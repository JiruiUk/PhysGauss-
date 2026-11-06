"""Single preregistered J7 memory experiment, serialized and opt-in."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from variants.staged_reconstruction.j7_long_memory import runtime as a
from variants.staged_reconstruction.i10_temporal_memory.core import build_bank,bind_memory,add_memory
from variants.staged_reconstruction.prepare_data import ROOT,DATA,read,sha
from experiments.staged_reconstruction.RubberDuck_joint_pilot import launch as old

def worker(phase,arm='temporal_memory',replicate=0):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=a.REPORT/f'{stamp}_{phase}_{arm}_{replicate}.log';path.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--enable-j7-memory','--phase',phase,'--arm',arm,'--replicate',str(replicate)]
    with path.open('x',encoding='utf-8') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW).returncode
    assert code==0,f'Failed log retained: {path}'
    return dict(log=path.relative_to(ROOT).as_posix(),sha256=sha(path))

def prepare():
    import torch
    import numpy as np
    from collections import Counter
    from output.short_clip_diagnostic.camera_math import w2c,intrinsic
    from variants.staged_reconstruction.i10_tracking.core import project_world
    r=a.install(True);assert not r.stopped() and not (a.REPORT/'source_freeze.json').exists()
    assert read(a.REPORT/'unit_tests.json')['passed'];torch.set_num_threads(4)
    p=read(a.BASE/'protocol.json');renderer,scene,opts,teacher,parent=r.load('temporal_memory')
    capture=lambda:deepcopy(r.cpu_values(dict(model=scene.state_dict(),optimizers=[o.state_dict() for o in opts],rng=a.memory.rng_state(),camera=parent['camera_stack'],graphs=parent['physics_graphs'])))
    before=capture();meta=read(DATA/'transforms_train.json')
    views=np.array([w2c(f['transform_matrix'],'metadata_cv') for f in meta['frames']]);k=intrinsic(meta['camera_angle_x'],'metadata',400,225)
    with a.memory.evidence_context(scene):xyz=np.stack([scene.dynamic_at(f['time']).get_xyz.cpu().numpy() for f in meta['frames']])
    r.assert_equivalent(before,capture(),0,0)
    path=ROOT/p['trajectories'];path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
    np.savez_compressed(path,xyz=xyz,views=views,k=k)
    bank=build_bank(read(ROOT/p['tracks'])['tracks'],read(ROOT/p['evidence']),p)
    uv,z=project_world(xyz,views,k)
    for m in bank['memories']:m['binding']=bind_memory(m,xyz,uv,z,p)
    bank.update(builder_source_sha256=sha(ROOT/'variants/staged_reconstruction/i10_temporal_memory/core.py'),
        parent_sha256=p['parent_sha256'],old_xyz_or_admission_reused=False,
        inputs_sha256={p[q]:sha(ROOT/p[q]) for q in ['tracks','evidence','evidence_review','trajectories']})
    r.atomic_json(a.REPORT/'bank.json',bank)
    a.memory.MEMORIES=[m for m in bank['memories'] if m['binding']['accepted']];a.memory.ASSET=dict(views=views,k=k)
    summary=a.memory.refresh(renderer,scene,True);r.assert_equivalent(before,capture(),0,0)
    passed=summary['active_memories']>=p['min_initial_memories'] and summary['active_rows']>=p['min_initial_distinct_rows']
    r.atomic_json(a.REPORT/'initial_health.json',dict(controller=a.memory.STATE,summary=summary,full_state_unchanged=True))
    r.atomic_json(a.REPORT/'initial_coverage.json',dict(passed=passed,summary=summary,
        original_tracklets=bank['original_track_count'],all_memories=len(bank['memories']),
        candidate_memories=sum(m['candidate'] for m in bank['memories']),geometry_bound=len(a.memory.MEMORIES),
        span_histogram=dict(Counter(len(m['frames']) for m in bank['memories'])),historical_coverage_failures_preserved=True))
    files=read(ROOT/'experiments/staged_reconstruction/RubberDuck_colorfree_joint/v1/source_freeze.json').copy()
    for name,h in files.items():assert sha(ROOT/name)==h,name
    paths=list((ROOT/'variants/staged_reconstruction/j7_long_memory').glob('*.py'))
    paths+=list((ROOT/'variants/staged_reconstruction/i10_temporal_memory').glob('*.py'))
    paths += [Path(__file__),a.BASE/'plan.md',a.BASE/'protocol.json',a.BASE/'preregistration.json',a.REPORT/'bank.json',a.REPORT/'initial_health.json',ROOT/'tests/test_j7_long_memory.py',ROOT/'tests/test_i10_temporal_memory.py']
    paths += [ROOT/p[q] for q in ['parent','tracks','evidence','evidence_review','trajectories']]
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'envs'):paths.append(path)
    for path in paths:files[path.relative_to(ROOT).as_posix()]=sha(path)
    r.atomic_json(a.REPORT/'source_freeze.json',files);r.verify()
    print('J8_INITIAL_QUALIFICATION',passed,summary,flush=True)
    assert passed,'No long native-admitted J7 memory: preserve evidence, do not relax or train'

def native():
    import torch
    from variants.staged_reconstruction.train_static import camera
    from variants.staged_reconstruction.rubber_joint.core import objective as baseline
    r=a.install(True);r.verify();torch.set_num_threads(4)
    renderer,scene,opts,teacher,parent=r.load('temporal_memory')
    before=deepcopy(r.cpu_values(r.snapshot(scene,opts,0,parent['camera_stack'],[])))
    assert a.memory.measure(renderer,scene)==[s['measurement'] for s in a.memory.STATE['memories']]
    r.assert_equivalent(before,r.snapshot(scene,opts,0,parent['camera_stack'],[]),0,0)
    flags=[p.requires_grad for p in scene.parameters()]
    try:
        with a.memory.evidence_context(scene):
            torch.rand(4,device='cuda');next(scene.parameters()).requires_grad_(False)
            raise ValueError('expected exception fixture')
    except ValueError:pass
    assert flags==[p.requires_grad for p in scene.parameters()]
    r.assert_equivalent(before,r.snapshot(scene,opts,0,parent['camera_stack'],[]),0,0)
    active=next(i for i,s in enumerate(a.memory.STATE['memories']) if s['status']=='active')
    index=next(q['index'] for q in a.memory.STATE['memories'][active]['measurement']['frames'] if not q['index']%2 and q['passed'])
    frame=a.memory.META['frames'][index];cam=camera(frame,a.memory.META,1600);target=r.frame_target(frame)
    loss,pkg,_,_=baseline(renderer,scene,teacher,cam,frame['time'],target,True)
    images={k:pkg[k].detach().clone() for k in ['render','alpha','depth']};loss.backward()
    grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in scene.named_parameters()}
    scene.zero_grad(set_to_none=True);del loss,pkg
    loss,pkg,_,_=a.objective(renderer,scene,teacher,cam,frame['time'],target)
    assert all(torch.equal(images[k],pkg[k]) for k in images);loss.backward()
    r.assert_equivalent(grads,{n:p.grad for n,p in scene.named_parameters()},1e-6,1e-5)
    r.assert_equivalent(before,r.snapshot(scene,opts,0,parent['camera_stack'],[]),0,0)
    scene.zero_grad(set_to_none=True);del loss,pkg,grads
    loss,pkg,_,_=a.objective(renderer,scene,teacher,cam,frame['time'],target,True)
    aux=loss-baseline(renderer,scene,teacher,cam,frame['time'],target,True)[0]
    assert a.LAST['endpoint_pairs']>0 and aux>0
    # Isolate the memory graph without a second renderer backward.
    scene.zero_grad(set_to_none=True)
    base=torch.zeros((),device='cuda',requires_grad=True)
    _,aux,counts=add_memory(base,lambda f:scene.dynamic_at(a.memory.META['frames'][f]['time']).get_xyz,
        a.memory.ASSET['views'],a.memory.ASSET['k'],index,a.memory.MEMORIES,a.memory.STATE['memories'],a.memory.CONFIG,True)
    aux.backward();norms={}
    for prefix in ['pools.0._xyz','grids.','networks.']:
        values=[p.grad for n,p in scene.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert values and all(torch.isfinite(v).all() for v in values)
        norms[prefix]=sum(float(v.abs().sum()) for v in values);assert norms[prefix]>0
    assert all(p.grad is None for n,p in scene.named_parameters() if n.startswith(('static.','physics.')))
    r.atomic_json(a.REPORT/'native_check.json',dict(passed=True,width=1600,flags_off_RGB_alpha_D_exact=True,
        base_J7_RGB_selfdepth_gradient_parity=True,independent_native_measurement_exact=True,
        no_mutation_exception_RNG_flags=True,memory_gradients=norms,isolated_memory_no_static_physics_gradient=True,
        keyframes=counts['keyframes'],endpoint_pairs=counts['endpoint_pairs']))
    r.verify();print('J8_NATIVE_PASS',norms,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-j7-memory',action='store_true')
    p.add_argument('--phase',choices=['prepare','native','qualify','smoke','recovery','train','pipeline'],default='prepare')
    p.add_argument('--arm',choices=['rgb_control','temporal_memory'],default='temporal_memory')
    p.add_argument('--replicate',type=int,choices=[0,1],default=0);args=p.parse_args()
    if not args.enable_j7_memory:raise SystemExit('J8 default off')
    r=a.install(True);old.worker=worker;assert not r.stopped()
    if args.phase=='prepare':prepare()
    elif args.phase=='native':native()
    elif args.phase=='qualify':old.qualify()
    elif args.phase=='smoke':r.train(args.arm,True)
    elif args.phase=='recovery':r.recovery_worker(args.arm,args.replicate)
    elif args.phase=='train':r.train(args.arm)
    else:old.pipeline()

if __name__=='__main__':main()
