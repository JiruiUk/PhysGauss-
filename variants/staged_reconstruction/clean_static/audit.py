"""Independent source, parent, row/moment and continuation-sampler audit."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from ..prepare_data import ROOT, read, write, sha
from ..adaptive_static.prepare import REPORT as PREVIOUS_REPORT, RUNS as PREVIOUS_RUNS
from ..adaptive_static.train import confidence_hash
from .prepare import REPORT, RUNS, verify


def freeze():
    assert not (REPORT/'source_freeze.json').exists(), 'Do not replace a frozen experiment'
    frozen = read(PREVIOUS_REPORT/'source_freeze.json')
    for p,expected in frozen.items(): assert sha(ROOT/p) == expected,p
    paths = list(Path(__file__).parent.glob('*.py'))+[ROOT/'tests/test_clean_static.py',
        REPORT/'protocol.json', REPORT/'discovery.json', REPORT/'candidate_review.json',
        REPORT/'structure_probe.json', REPORT/'test_receipt.json',
        ROOT/read(REPORT/'protocol.json')['parent']]+list(RUNS.glob('*.npz'))
    for p in paths: frozen[p.relative_to(ROOT).as_posix()] = sha(p)
    write(REPORT/'source_freeze.json',frozen); verify()


def equal_rng(a,b,include_python=True):
    if include_python: assert a['rng_python'] == b['rng_python']
    assert a['rng_numpy'][0] == b['rng_numpy'][0] and np.array_equal(a['rng_numpy'][1],b['rng_numpy'][1])
    assert a['rng_numpy'][2:] == b['rng_numpy'][2:]
    assert torch.equal(a['rng_torch'],b['rng_torch'])
    assert len(a['rng_cuda']) == len(b['rng_cuda']) and all(torch.equal(x,y) for x,y in zip(a['rng_cuda'],b['rng_cuda']))


def audit(arm,smoke,n):
    protocol = read(REPORT/'protocol.json'); parent = torch.load(ROOT/protocol['parent'],map_location='cpu',weights_only=False)
    directory = RUNS/('smoke' if smoke else 'formal')/arm; path = directory/'checkpoint.pt'
    state = torch.load(path,map_location='cpu',weights_only=False)
    initial = torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
    assert state['iteration'] == n and initial['iteration'] == 12000
    assert sha(path) == read(directory/'checkpoint.json')['sha256']
    for s in [state,initial]:
        assert s['arm'] == arm and s['protocol_sha256'] == sha(REPORT/'protocol.json')
        assert s['source_manifest_sha256'] == sha(REPORT/'source_freeze.json')
        assert s['parent_sha256'] == sha(ROOT/protocol['parent']) == protocol['parent_sha256']
        assert s['static_only'] and s['dynamic_parameters'] == 0
        assert all(v.isfinite().all() for v in s['model'].values())
        assert all(v.isfinite().all() for row in s['optimizer']['state'].values() for v in row.values() if torch.is_tensor(v))
        assert confidence_hash(s['confidence']) == s['confidence_sha256']
        assert s['parent_confidence_history'] == parent['confidence_history']
    with np.load(RUNS/'parent_evidence.npz') as d: rejected = d['quarantine'].copy()
    keep = torch.from_numpy(~rejected if arm == 'visibility' else np.ones(len(rejected),bool))
    for key,value in parent['model'].items(): assert torch.equal(initial['model'][key],value[keep]),key
    for a,b in zip(initial['optimizer']['param_groups'],parent['optimizer']['param_groups']): assert a == b
    for i,row in parent['optimizer']['state'].items():
        for key,value in row.items():
            expected = value[keep] if value.ndim else value
            assert torch.equal(initial['optimizer']['state'][i][key],expected),(i,key)
    removed = parent['model']['point_ids'][~keep].tolist()
    assert initial['removed_ids'] == state['removed_ids'] == removed
    assert state['model']['point_ids'].tolist() == initial['model']['point_ids'].tolist()
    assert torch.equal(state['model']['initial_xyz'],initial['model']['initial_xyz'])
    assert torch.equal(state['model']['born_at'],initial['model']['born_at'])
    counters = [int(row['step']) for row in state['optimizer']['state'].values()]
    assert len(counters) == 6 and counters == [n]*6
    equal_rng(initial,parent); assert initial['camera_stack'] == parent['camera_stack']
    contract = read(directory/'sampling_contract.json')
    assert contract['frames'] == read(PREVIOUS_RUNS/'formal/geometry/sampling_contract.json')['frames']
    count = len(contract['frames']); rng = random.Random(); rng.setstate(parent['rng_python'])
    stack,sequence = parent['camera_stack'].copy(),[]
    for _ in range(n-12000):
        if not stack: stack = rng.sample(range(count),count)
        sequence.append(stack.pop())
    actual = [json.loads(line) for line in (directory/'sampling.jsonl').read_text().splitlines()]
    assert [r['step'] for r in actual] == list(range(12001,n+1))
    assert [r['frame_index'] for r in actual] == sequence
    assert state['camera_stack'] == stack and state['rng_python'] == rng.getstate()
    equal_rng(initial,state,include_python=False)
    with np.load((RUNS if arm == 'visibility' else PREVIOUS_RUNS)/'geometric_confidence.npz') as geometry:
        for key,value in state['confidence'].items():
            assert np.isfinite(value).all() and (value >= 0).all() and (value <= geometry[key]+1e-6).all()
            assert np.array_equal(initial['confidence'][key],np.minimum(parent['confidence'][key],geometry[key]))
    interval = 10 if smoke else protocol['confidence_interval']
    rounds = list(range(12000+interval,n+1,interval))
    assert [e['step'] for e in state['confidence_history']] == rounds
    assert state['confidence_history'][-1]['confidence_sha256'] == state['confidence_sha256']
    if smoke and n == 12080:
        before = read(REPORT/f'smoke12040_{arm}.json')
        assert before['passed'] and before['sampler'] == sequence[:40]
        assert before['confidence_sha256'] == state['confidence_history'][3]['confidence_sha256']
        assert before['checkpoint_sha256'] != sha(path)
    result = {'passed':True,'arm':arm,'iteration':n,'checkpoint_sha256':sha(path),'optimizer_steps':counters,
        'point_count':len(state['model']['point_ids']),'quarantined_count':len(removed),'sampler':sequence if smoke else None,
        'confidence_sha256':state['confidence_sha256'],'confidence_rounds':rounds,
        'full_parent_model_Adam_RNG_sampler_verified':True,'unchanged_topology_verified':True,
        'nonzero_supervision_steps':sum(r['supervised_pixels'] > 0 for r in actual),
        'source_and_historical_checkpoints_preserved':True}
    write(REPORT/(f'smoke{n}_{arm}.json' if smoke else f'formal_{arm}_audit.json'),result); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--step',type=int,default=18000)
    parser.add_argument('--arm',choices=['control','visibility']); args = parser.parse_args()
    if args.freeze: freeze(); print('CLEAN_STATIC_FROZEN',flush=True); return
    verify(); result = audit(args.arm,args.smoke,args.step); print('CLEAN_STATIC_AUDITED',args.arm,args.step,result['passed'],flush=True)
    if args.smoke and args.step == 12080:
        other = REPORT/f'smoke12080_{"visibility" if args.arm == "control" else "control"}.json'
        if other.exists():
            write(REPORT/'smoke_verification.json',{'passed':True,'independent_process12040_to12080_both_arms':True,
                'full12k_parent_preserved_including_Adam_RNG_and_camera_stack':True,
                'source_manifest_sha256':sha(REPORT/'source_freeze.json')})


if __name__ == '__main__': main()
