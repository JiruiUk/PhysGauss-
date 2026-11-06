"""Preregistered single temporal-memory qualification and matched experiment."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from variants.staged_reconstruction.i10_temporal_memory import runtime as a
from variants.staged_reconstruction.i10_temporal_memory.core import build_bank,bind_memory,add_memory
from variants.staged_reconstruction.i10_tracking.io import ROOT,DATA,read,sha
from experiments.staged_reconstruction.RubberDuck_joint_pilot import launch as old

def worker(phase,arm='temporal_memory',replicate=0):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=a.REPORT/f'{stamp}_{phase}_{arm}_{replicate}.log';path.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--enable-temporal-memory','--phase',phase,'--arm',arm,'--replicate',str(replicate)]
    with path.open('x',encoding='utf-8') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,f'Failed log retained: {path}'
    return dict(log=path.relative_to(ROOT).as_posix(),sha256=sha(path))

def prepare():
    import torch
    import numpy as np
    from variants.staged_reconstruction.i10_tracking.core import project_world
    r=a.install(True);assert not r.stopped() and not (a.REPORT/'source_freeze.json').exists()
    assert read(a.REPORT/'unit_tests.json')['passed'];torch.set_num_threads(4)
    p=read(a.BASE/'protocol.json');bank=read(a.REPORT/'bank.json')
    assert bank['builder_source_sha256']==sha(ROOT/'variants/staged_reconstruction/i10_temporal_memory/core.py')
    tracks=read(ROOT/p['tracks'])['tracks'];evidence=read(ROOT/p['evidence'])
    replay=build_bank(tracks,evidence,p)
    with np.load(ROOT/p['trajectories']) as asset:
        projected,depth=project_world(asset['xyz'],asset['views'],asset['k'])
        for m in replay['memories']:m['binding']=bind_memory(m,asset['xyz'],projected,depth,p)
    assert all(replay[k]==bank[k] for k in replay)
    assert not read(ROOT/p['evidence_review'])['pending_review']
    renderer,scene,opts,_,parent=r.load('temporal_memory')
    capture=lambda:r.cpu_values(dict(model=scene.state_dict(),optimizers=[o.state_dict() for o in opts],rng=a.rng_state(),camera=parent['camera_stack'],graphs=parent['physics_graphs']))
    before=deepcopy(capture());summary=a.refresh(renderer,scene,True);r.assert_equivalent(before,capture(),0,0)
    passed=summary['active_memories']>=p['min_initial_memories'] and summary['active_rows']>=p['min_initial_distinct_rows']
    r.atomic_json(a.REPORT/'initial_health.json',dict(controller=a.STATE,summary=summary,full_state_unchanged=True))
    r.atomic_json(a.REPORT/'initial_coverage.json',dict(passed=passed,summary=summary,historical_global_coverage_failures_preserved=True))
    r.atomic_json(a.REPORT/'input_replay.json',dict(passed=True,all730_tracklets_and_links_replayed=True,all_binding_scores_replayed=True,
        unknown_tracks_and_unbound_memories_retained=True,no_model_twoAdam_RNG_camera_physics_mutation=True))
    files={};inherited=read(ROOT/'experiments/staged_reconstruction/RubberDuck_I10_track_neighborhood/visible/v1/source_freeze.json')
    for name,h in inherited.get('files',inherited).items():assert sha(ROOT/name)==h,name;files[name]=h
    paths=list((ROOT/'variants/staged_reconstruction/i10_temporal_memory').glob('*.py'))
    paths+=list(a.BASE.glob('*.py'))+[a.BASE/'plan.md',a.BASE/'protocol.json',a.BASE/'preregistration.json',
        a.REPORT/'bank.json',a.REPORT/'initial_health.json',ROOT/'tests/test_i10_temporal_memory.py',ROOT/p['tracks'],ROOT/p['evidence_review']]
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'envs'):paths.append(path)
    for path in paths:files[path.relative_to(ROOT).as_posix()]=sha(path)
    r.atomic_json(a.REPORT/'source_freeze.json',files);r.verify()
    print('L4_INITIAL_QUALIFICATION',passed,summary,flush=True)
    assert passed,'No long native-admitted memory: stop, preserve report, do not relax or train'

def native():
    import torch
    from variants.staged_reconstruction.train_static import camera
    from variants.staged_reconstruction.rubber_joint.core import objective as baseline
    r=a.install(True);r.verify();torch.set_num_threads(4)
    renderer,scene,opts,_,parent=r.load('temporal_memory')
    before=deepcopy(r.cpu_values(r.snapshot(scene,opts,0,parent['camera_stack'],[])))
    assert a.measure(renderer,scene)==[s['measurement'] for s in a.STATE['memories']]
    r.assert_equivalent(before,r.snapshot(scene,opts,0,parent['camera_stack'],[]),0,0)
    try:
        with a.evidence_context(scene):
            torch.rand(4,device='cuda');next(scene.parameters()).requires_grad_(False)
            raise ValueError('expected exception fixture')
    except ValueError:pass
    r.assert_equivalent(before,r.snapshot(scene,opts,0,parent['camera_stack'],[]),0,0)
    active=next(i for i,s in enumerate(a.STATE['memories']) if s['status']=='active')
    index=next(q['index'] for q in a.STATE['memories'][active]['measurement']['frames'] if not q['index']%2 and q['passed'])
    frame=a.META['frames'][index];cam=camera(frame,a.META,1600);target=r.frame_target(frame)
    loss,pkg,_,_=baseline(renderer,scene,None,cam,frame['time'],target)
    disabled,_,_=add_memory(loss,*([None]*7),enabled=False);assert disabled is loss
    images={k:pkg[k].detach().clone() for k in ['render','alpha','depth']};disabled.backward()
    grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in scene.named_parameters()}
    scene.zero_grad(set_to_none=True);del loss,pkg,disabled
    loss,pkg,_,_=baseline(renderer,scene,None,cam,frame['time'],target);loss.backward()
    assert all(torch.equal(images[k],pkg[k]) for k in images)
    r.assert_equivalent(grads,{n:p.grad for n,p in scene.named_parameters()},1e-6,1e-5)
    scene.zero_grad(set_to_none=True);del loss,pkg,grads
    loss,pkg,rgb,aux=a.objective(renderer,scene,None,cam,frame['time'],target)
    assert all(torch.equal(images[k],pkg[k]) for k in images);assert aux>0 and a.LAST['endpoint_pairs']>0
    aux.backward();norms={}
    for prefix in ['pools.0._xyz','grids.','networks.']:
        values=[p.grad for n,p in scene.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert values and all(torch.isfinite(v).all() for v in values)
        norms[prefix]=sum(float(v.abs().sum()) for v in values);assert norms[prefix]>0
    assert all(p.grad is None for n,p in scene.named_parameters() if n.startswith(('static.','physics.')))
    r.atomic_json(a.REPORT/'native_check.json',dict(passed=True,width=1600,flags_off_RGB_alpha_D_exact=True,
        flags_off_gradients_original_tolerance=True,all_memory_measurements_independent_process_exact=True,
        no_mutation_exception_rng_flags=True,joint_temporal_gradients=norms,static_physics_frozen=True,
        keyframes=a.LAST['keyframes'],endpoint_pairs=a.LAST['endpoint_pairs']))
    r.verify();print('L4_NATIVE_PASS',norms,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-temporal-memory',action='store_true')
    parser.add_argument('--phase',choices=['prepare','native','qualify','smoke','recovery','train','pipeline'],default='prepare')
    parser.add_argument('--arm',choices=['rgb_control','temporal_memory'],default='temporal_memory')
    parser.add_argument('--replicate',type=int,choices=[0,1],default=0);args=parser.parse_args()
    if not args.enable_temporal_memory:raise SystemExit('Temporal memory is default off')
    r=a.install(True);old.worker=worker;assert not r.stopped()
    if args.phase=='prepare':prepare()
    elif args.phase=='native':native()
    elif args.phase=='qualify':old.qualify()
    elif args.phase=='smoke':r.train(args.arm,True)
    elif args.phase=='recovery':r.recovery_worker(args.arm,args.replicate)
    elif args.phase=='train':r.train(args.arm)
    else:old.pipeline()

if __name__=='__main__':main()
