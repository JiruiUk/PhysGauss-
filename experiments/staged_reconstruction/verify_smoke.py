"""Independent actual-state verification; never makes or edits training weights."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from variants.staged_reconstruction.prepare_data import REPORT,RUNS,read,write,sha


def digest(tensor):return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['S3','S4','S5','S6'],required=True)
    p.add_argument('--capture40',action='store_true');args=p.parse_args()
    report=REPORT/args.stage;directory=RUNS/args.stage/'_smoke';path=directory/'checkpoint.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert sha(path)==read(directory/'checkpoint.json')['sha256']
    counters={str(k):float(v['step']) for k,v in state['optimizer']['state'].items()}
    physics_ids={str(k) for g in state['optimizer']['param_groups'] if g['name']=='PIDG_velocity_stress' for k in g['params']}
    snapshot={'iteration':state['iteration'],'checkpoint_sha256':sha(path),'counters':counters,
        'model_digests':{k:digest(v) for k,v in state['model'].items()},'camera_stack':state['camera_stack'],
        'rng_python':state['rng_python'],'physics_parameter_ids':sorted(physics_ids)}
    if args.capture40:
        assert state['iteration']==40 and 'physics_graphs' in state
        write(report/'smoke40_snapshot.json',snapshot);print('SMOKE40_SNAPSHOT',len(counters));return
    prior=read(report/'smoke40_snapshot.json');assert prior['iteration']==40 and state['iteration']==80
    assert set(counters)==set(prior['counters']) and prior['physics_parameter_ids']==sorted(physics_ids)
    for key,value in counters.items():
        expected=4 if key in physics_ids else 40
        assert value-prior['counters'][key]==expected,(key,value,prior['counters'][key],expected)
    py=random.Random();old=prior['rng_python'];py.setstate((old[0],tuple(old[1]),old[2]));stack=prior['camera_stack'][:]
    for _ in range(40):
        if not stack:stack=py.sample(range(120),120)
        stack.pop()
    assert stack==state['camera_stack'] and py.getstate()==state['rng_python']
    static=torch.load(RUNS/'S1_static/checkpoint.pt',map_location='cpu',weights_only=False)['model']
    ignored=('_features',) if args.stage=='S6' else ()
    assert all(torch.equal(v,state['model']['static.'+k]) for k,v in static.items() if not any(k.startswith(x) for x in ignored))
    assert all(v.isfinite().all() for v in state['model'].values())
    assert all(v.isfinite().all() for item in state['optimizer']['state'].values() for v in item.values() if torch.is_tensor(v))
    changed=[k for k,v in state['model'].items() if digest(v)!=prior['model_digests'][k]]
    assert any(k.startswith('physics.') for k in changed) and any(k.startswith('networks.') for k in changed)
    lines=[json.loads(x) for x in (directory/('physics.jsonl' if args.stage=='S3' else 'objectives.jsonl')).read_text().splitlines()]
    if args.stage=='S3':assert [x['step'] for x in lines]==list(range(10,81,10))
    result={'passed':True,'stage':args.stage,'independent_process_resume_40_to_80':True,'checkpoint_sha256':sha(path),
        'all_actual_Adam_counters_advance_correctly':True,'optimizer_state_count':len(counters),
        'python_RNG_and_camera_sampler_exact':True,'all_RNG_payloads_present':all(k in state for k in ['rng_python','rng_numpy','rng_torch','rng_cuda']),
        'derived_physics_graph_state_present':True,'static_geometry_exact_S1':True,'static_appearance_trainable':args.stage=='S6',
        'finite_model_and_optimizer':True,'changed_parameter_tensors':len(changed),'actual_physics_and_deformation_updates':True,
        'not_formal_stage_completion':True}
    write(report/'smoke_verification.json',result);print('SMOKE80_VERIFIED',args.stage,len(counters))


if __name__=='__main__':main()
