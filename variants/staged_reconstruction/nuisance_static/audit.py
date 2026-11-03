"""Freeze and independently replay synthetic state, sampler and per-view Adam."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from ..prepare_data import ROOT,read,write,sha
from ..clean_static.prepare import REPORT as PREVIOUS_REPORT
from .prepare import REPORT,RUNS,verify


def freeze():
    assert not (REPORT/'source_freeze.json').exists(), 'Never overwrite a frozen experiment'
    frozen = read(PREVIOUS_REPORT/'source_freeze.json')
    for p,h in frozen.items(): assert sha(ROOT/p) == h,p
    protocol = read(REPORT.parent/'protocol.json'); parent = ROOT/protocol['real_parent']
    assert sha(parent) == protocol['real_parent_sha256']; frozen[protocol['real_parent']] = sha(parent)
    paths = list(Path(__file__).parent.glob('*.py'))+[ROOT/'tests/test_nuisance_static.py',
        REPORT.parent/'protocol.json',REPORT/'preparation.json',REPORT/'test_receipt.json',REPORT/'native_check.json']
    paths += list(RUNS.glob('*/*.npz'))
    for p in paths: frozen[p.relative_to(ROOT).as_posix()] = sha(p)
    frozen.update(read(REPORT/'preparation.json')['inputs'])
    write(REPORT/'source_freeze.json',frozen); verify()


def audit(arm,case,smoke,n):
    directory = RUNS/('smoke' if smoke else 'formal')/case/arm
    state = torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    initial = torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
    assert state['iteration'] == n and initial['iteration'] == 0
    assert sha(directory/'checkpoint.pt') == read(directory/'checkpoint.json')['sha256']
    assert state['arm'] == arm and state['case'] == case
    assert state['protocol_sha256'] == sha(REPORT.parent/'protocol.json')
    assert state['source_manifest_sha256'] == sha(REPORT/'source_freeze.json')
    assert state['physical_dynamic_parameters'] == 0 and state['point_ids_fixed']
    assert all(v.isfinite().all() for v in state['model'].values())
    assert torch.equal(state['model']['point_ids'],initial['model']['point_ids'])
    assert torch.equal(state['model']['initial_xyz'],initial['model']['initial_xyz'])
    counters = [int(s['step']) for s in state['optimizer']['state'].values()]
    assert len(counters) == 6 and counters == [n]*6
    count = len(state['visits']); rng = random.Random(); rng.setstate(initial['rng_python'])
    stack,sequence,visits = [],[],[0]*count
    for _ in range(n):
        if not stack: stack = rng.sample(range(count),count)
        index = stack.pop(); sequence.append(index); visits[index] += 1
    records = [json.loads(line) for line in (directory/'sampling.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records] == list(range(1,n+1))
    assert [r['view'] for r in records] == sequence and visits == state['visits']
    assert state['camera_stack'] == stack and state['rng_python'] == rng.getstate()
    assert initial['rng_numpy'][0] == state['rng_numpy'][0]
    assert np.array_equal(initial['rng_numpy'][1],state['rng_numpy'][1]) and initial['rng_numpy'][2:] == state['rng_numpy'][2:]
    assert torch.equal(initial['rng_torch'],state['rng_torch'])
    assert all(torch.equal(a,b) for a,b in zip(initial['rng_cuda'],state['rng_cuda']))
    if arm == 'full_RGB_nuisance':
        assert len(state['nuisance_optimizers']) == count
        assert all(v.isfinite().all() for v in state['nuisance'].values())
        for i,(optimizer,seen) in enumerate(zip(state['nuisance_optimizers'],visits)):
            steps = [int(s['step']) for s in optimizer['state'].values()]
            assert steps == [seen]*4 if seen else not steps
            if not seen:
                for key,value in initial['nuisance'].items():
                    if key.startswith(str(i)+'.'): assert torch.equal(value,state['nuisance'][key]),key
            assert all(v.isfinite().all() for row in optimizer['state'].values() for v in row.values() if torch.is_tensor(v))
    else: assert state['nuisance'] is None and not state['nuisance_optimizers']
    other = RUNS/('smoke' if smoke else 'formal')/case/('full_RGB_control' if arm == 'full_RGB_nuisance' else 'full_RGB_nuisance')/'initial.pt'
    if other.exists():
        a = torch.load(other,map_location='cpu',weights_only=False)
        assert all(torch.equal(a['model'][k],v) for k,v in initial['model'].items())
        assert a['rng_python'] == initial['rng_python']
    if smoke and n == 80:
        before = read(REPORT/f'smoke40_{case}_{arm}.json')
        assert before['sampler'] == sequence[:40] and before['checkpoint_sha256'] != sha(directory/'checkpoint.pt')
        assert before['visits'] == [sequence[:40].count(i) for i in range(count)]
    result = {'passed':True,'arm':arm,'case':case,'iteration':n,'checkpoint_sha256':sha(directory/'checkpoint.pt'),
        'static_optimizer_steps':counters,'visits':visits,'sampler':sequence if smoke else None,
        'per_view_only_optimizer_updates_verified':True,'unchanged_unvisited_views_verified':True,
        'full_Adam_RNG_sampler_and_topology_verified':True}
    write(REPORT/(f'smoke{n}_{case}_{arm}.json' if smoke else f'formal_{case}_{arm}_audit.json'),result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--arm',choices=['full_RGB_control','full_RGB_nuisance']); parser.add_argument('--case',default='diffuse')
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--step',type=int,default=6000); args = parser.parse_args()
    if args.freeze: freeze(); print('NUISANCE_FROZEN',flush=True); return
    verify(); audit(args.arm,args.case,args.smoke,args.step); print('NUISANCE_AUDITED',args.case,args.arm,args.step,flush=True)
    if args.smoke and args.step == 80:
        arms = ['full_RGB_control','full_RGB_nuisance']
        if all((REPORT/f'smoke80_diffuse_{a}.json').exists() for a in arms):
            a,b = [read(REPORT/f'smoke80_diffuse_{arm}.json') for arm in arms]; assert a['sampler'] == b['sampler']
            write(REPORT/'smoke_verification.json',{'passed':True,'independent_process40_to80_both_arms':True,
                'matched_initial_static_and_camera_order':True,'source_manifest_sha256':sha(REPORT/'source_freeze.json')})


if __name__ == '__main__': main()
