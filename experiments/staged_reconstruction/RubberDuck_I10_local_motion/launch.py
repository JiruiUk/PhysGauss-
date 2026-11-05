"""One qualified full-state local-motion comparison; old jobs stay paused."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from variants.staged_reconstruction.i10_local_motion import runtime as a
from variants.staged_reconstruction.i10_local_motion.core import support,add_local
from variants.staged_reconstruction.i10_track_train.core import add_tracks
from variants.staged_reconstruction.i10_tracking.io import ROOT,DATA,read,sha
from experiments.staged_reconstruction.RubberDuck_joint_pilot import launch as old


def worker(phase,arm='local_motion',replicate=0):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=a.REPORT/f'{stamp}_{phase}_{arm}_{replicate}.log';path.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--enable-local-motion','--phase',phase,'--arm',arm,'--replicate',str(replicate)]
    with path.open('x',encoding='utf-8') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,f'Failed log preserved: {path}'
    return dict(log=path.relative_to(ROOT).as_posix(),sha256=sha(path))


def prepare():
    import numpy as np
    import torch
    r=a.install(True);assert not r.stopped() and not (a.REPORT/'source_freeze.json').exists()
    torch.set_num_threads(4);assert read(a.REPORT/'unit_tests.json')['passed']
    p=read(a.BASE/'protocol.json');evidence=read(ROOT/p['evidence'])
    review=read(ROOT/p['evidence_review']);assert not review['pending_review']
    assert review['status']=='completed_not_admitted' and not review['neighborhood_gate_passed']
    _,scene,opts,_,parent=r.load('local_motion')
    with np.load(ROOT/p['trajectories']) as f:obs,pairs,summary=support(evidence,f['xyz'])
    assert summary['accepted_tracks']==122 and summary['qualified_graph_tracks']==116
    r.atomic_json(a.REPORT/'support.json',dict(**summary,parent_sha256=p['parent_sha256'],
        evidence_sha256=sha(ROOT/p['evidence']),new_scope_explicitly_supported_windows_only=True,
        original_T3_T4_global_gate_still_failed=True,static_and_physics_frozen=True,
        model_twoAdam_RNG_camera_graphs_exact_start=True,legacy_color_history=True))
    files={}
    for receipt in [ROOT/'experiments/staged_reconstruction/RubberDuck_I10_track_neighborhood/visible/v1/source_freeze.json']:
        values=read(receipt)
        for name,h in values.get('files',values).items():assert sha(ROOT/name)==h,name;files[name]=h
    paths=list((ROOT/'variants/staged_reconstruction/i10_local_motion').glob('*.py'))
    paths += [a.BASE/'plan.md',a.BASE/'protocol.json',a.BASE/'preregistration.json',Path(__file__),a.REPORT/'support.json',
              ROOT/'tests/test_i10_local_motion.py',ROOT/'tests/test_i10_track_train.py',ROOT/p['evidence_review']]
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'envs'):paths.append(path)
    for path in paths:files[path.relative_to(ROOT).as_posix()]=sha(path)
    r.atomic_json(a.REPORT/'source_freeze.json',files);r.verify()
    print('L1_SUPPORTED_WINDOWS_AND_SOURCE_FROZEN',len(files),summary,flush=True)


def native():
    import torch
    from variants.staged_reconstruction.train_static import camera
    from variants.staged_reconstruction.rubber_joint.core import objective as baseline
    r=a.install(True);r.verify();torch.set_num_threads(4)
    renderer,scene,opts,_,_=r.load('audited_tracks')
    meta=read(DATA/'transforms_train.json');index=min(a.PAIRS);frame=meta['frames'][index]
    cam=camera(frame,meta,1600);target=r.frame_target(frame)
    base,pkg,_,_=baseline(renderer,scene,None,cam,frame['time'],target)
    disabled,zero=add_tracks(base,None,None,None,None,False)
    disabled,zero2=add_local(disabled,None,None,False)
    assert disabled is base and zero==zero2==0
    images={k:pkg[k].detach().clone() for k in ['render','alpha','depth']}
    disabled.backward();old_grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in scene.named_parameters()}
    scene.zero_grad(set_to_none=True);del base,pkg,disabled
    base,pkg,_,_=baseline(renderer,scene,None,cam,frame['time'],target)
    base.backward();r.assert_equivalent(old_grads,{n:p.grad for n,p in scene.named_parameters()},1e-6,1e-5)
    assert all(torch.equal(images[k],pkg[k]) for k in images)
    scene.zero_grad(set_to_none=True);del base,pkg,old_grads,images
    loss,pkg,_,_=r.objective(renderer,scene,None,cam,frame['time'],target)
    loss.backward();control={n:None if p.grad is None else p.grad.detach().clone() for n,p in scene.named_parameters()}
    images={k:pkg[k].detach().clone() for k in ['render','alpha','depth']}
    scene.zero_grad(set_to_none=True);del loss,pkg
    a.ARM='local_motion';loss,pkg,_,_=r.objective(renderer,scene,None,cam,frame['time'],target)
    assert a.LAST['weighted_local_loss']<1e-12
    assert all(torch.equal(images[k],pkg[k]) for k in images)
    loss.backward();r.assert_equivalent(control,{n:p.grad for n,p in scene.named_parameters()},1e-6,1e-5)
    scene.zero_grad(set_to_none=True);del loss,pkg,control,images
    world=scene.dynamic_at(frame['time']).get_xyz
    perturb=torch.zeros_like(world);perturb[a.PAIRS[index][0][0],0]=.003
    isolated,term=add_local(world.sum()*0,world+perturb,a.PAIRS[index],True,.01,.05)
    isolated.backward();norms={}
    for prefix in ['pools.0._xyz','grids.','networks.']:
        grads=[p.grad for n,p in scene.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        norms[prefix]=sum(float(g.abs().sum()) for g in grads);assert norms[prefix]>0
    assert all(p.grad is None for n,p in scene.named_parameters() if n.startswith(('static.','physics.')))
    assert all(p.grad is None for n,p in scene.named_parameters() if n.startswith('pools.0.') and n!='pools.0._xyz')
    r.atomic_json(a.REPORT/'native_check.json',dict(passed=True,width=1600,
        disabled_RGB_alpha_D_exact=True,disabled_native_gradients_original_tolerance=True,
        initial_local_zero_and_native_gradients_equal_tracks=True,
        isolated_local_canonical_grid_deform_finite_nonzero=norms,static_physics_frozen=True,
        optimizer_updates=0,isolated_local_loss=float(term.detach())))
    r.verify();print('L1_NATIVE_FLAGS_OFF_AND_LOCAL_GRADIENTS_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-local-motion',action='store_true')
    parser.add_argument('--phase',choices=['prepare','native','qualify','smoke','recovery','train','pipeline'],default='prepare')
    parser.add_argument('--arm',choices=['audited_tracks','local_motion'],default='local_motion')
    parser.add_argument('--replicate',type=int,choices=[0,1],default=0);args=parser.parse_args()
    if not args.enable_local_motion:raise SystemExit('Local motion default off')
    r=a.install(True);old.worker=worker;assert not r.stopped()
    if args.phase=='prepare':prepare()
    elif args.phase=='native':native()
    elif args.phase=='qualify':old.qualify()
    elif args.phase=='smoke':r.train(args.arm,True)
    elif args.phase=='recovery':r.recovery_worker(args.arm,args.replicate)
    elif args.phase=='train':r.train(args.arm)
    else:old.pipeline()


if __name__=='__main__':main()
