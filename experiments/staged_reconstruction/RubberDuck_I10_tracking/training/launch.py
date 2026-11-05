"""One opt-in matched RGB/3D-track continuation; no paused task is resumed."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from variants.staged_reconstruction.i10_track_train import runtime as a
from variants.staged_reconstruction.i10_tracking.io import ROOT,DATA,read,sha
from experiments.staged_reconstruction.RubberDuck_joint_pilot import launch as old


def worker(phase,arm='rgb_tracks',replicate=0):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=a.REPORT/f'{stamp}_{phase}_{arm}_{replicate}.log';path.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--enable-track-training','--phase',phase,'--arm',arm,'--replicate',str(replicate)]
    with path.open('x',encoding='utf-8') as f:
        code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,f'Failed log retained: {path}'
    return dict(log=path.relative_to(ROOT).as_posix(),sha256=sha(path))


def prepare():
    import torch
    from variants.staged_reconstruction.i10_track_train.core import observations
    r=a.install(True);assert not r.stopped() and not (a.REPORT/'source_freeze.json').exists()
    torch.set_num_threads(4)
    assert read(a.REPORT/'unit_tests.json')['passed']
    evidence_base=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_tracking/dense/v1'
    assert read(evidence_base/'final_review.json')['evidence_gate_passed']
    for receipt in ['independent_process_replay.json','tracker_replay.json']:assert read(evidence_base/receipt)['passed']
    _,scene,opts,_,parent=r.load('rgb_tracks')
    obs=observations(read(ROOT/read(a.BASE/'protocol.json')['evidence']))
    r.atomic_json(a.REPORT/'parent_audit.json',dict(parent_sha256=read(a.BASE/'protocol.json')['parent_sha256'],
        static_points=len(scene.static.get_xyz),dynamic_points=len(scene.pools[0].get_xyz),
        exact_model_two_Adam_RNG_sampler_graphs=True,legacy_color_history=True,new_color_reference=False,
        observations=sum(len(v) for v in obs.values()),covered_train_frames=len(obs),
        observation_counts_by_frame={str(i):len(obs.get(i,[])) for i in range(120)},optimizer_updates=0))
    files={}
    for receipt in [ROOT/'experiments/staged_reconstruction/RubberDuck_I10_joint/v1/source_freeze.json',evidence_base/'source_freeze.json']:
        content=read(receipt)
        for name,h in content.get('files',content).items():assert sha(ROOT/name)==h,name;files[name]=h
    paths=[a.BASE/'protocol.json',a.BASE/'plan.md',a.BASE/'preregistration.json',Path(__file__),ROOT/'tests/test_i10_track_train.py',
        evidence_base/'bindings.json',evidence_base/'final_review.json',evidence_base/'independent_process_replay.json',evidence_base/'tracker_replay.json']
    paths+=list((ROOT/'variants/staged_reconstruction/i10_track_train').glob('*.py'))
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'envs'):paths.append(path)
    for path in paths:files[path.relative_to(ROOT).as_posix()]=sha(path)
    r.atomic_json(a.REPORT/'source_freeze.json',files);r.verify()
    print('T2_PARENT_OBSERVATIONS_AND_SOURCE_FROZEN',len(files),len(obs),flush=True)


def native():
    import torch
    from variants.staged_reconstruction.train_static import camera
    from variants.staged_reconstruction.rubber_joint.core import objective as baseline
    from variants.staged_reconstruction.i10_track_train.core import add_tracks
    r=a.install(True);r.verify();torch.set_num_threads(4)
    renderer,scene,opts,_,_=r.load('rgb_only')
    meta=read(DATA/'transforms_train.json');frame=meta['frames'][0];cam=camera(frame,meta,1600);target=r.frame_target(frame)
    loss,pkg,_,_=baseline(renderer,scene,None,cam,frame['time'],target)
    native={k:pkg[k].detach().clone() for k in ['render','alpha','depth']}
    loss.backward();grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in scene.named_parameters()}
    scene.zero_grad(set_to_none=True);del loss,pkg
    loss,pkg,_,track=r.objective(renderer,scene,None,cam,frame['time'],target)
    assert track.item()==0 and all(torch.equal(native[k],pkg[k]) for k in native)
    loss.backward();r.assert_equivalent(grads,{n:p.grad for n,p in scene.named_parameters()},1e-6,1e-5)
    scene.zero_grad(set_to_none=True);del loss,pkg,native,grads
    isolated,term=add_tracks(scene.pools[0]._xyz.sum()*0,scene,cam,frame['time'],a.OBS[0],True,.01)
    isolated.backward();norms={}
    for prefix in ['pools.0._xyz','grids.','networks.']:
        values=[p.grad for n,p in scene.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert values and all(torch.isfinite(g).all() for g in values),prefix
        norms[prefix]=sum(float(g.abs().sum()) for g in values);assert norms[prefix]>0
    assert all(p.grad is None for n,p in scene.named_parameters() if n.startswith(('static.','physics.')))
    assert all(p.grad is None for n,p in scene.named_parameters() if n.startswith('pools.0.') and n!='pools.0._xyz')
    r.verify();r.atomic_json(a.REPORT/'native_check.json',dict(passed=True,width=1600,
        default_off_RGB_alpha_D_exact=True,flags_off_gradients_original_tolerance=True,
        isolated_track_canonical_grid_deform_finite_nonzero=norms,static_and_physics_frozen=True,
        track_loss=float(term.detach()),optimizer_updates=0))
    print('T2_NATIVE_FLAGS_OFF_AND_3D_GRADIENT_PASS',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-track-training',action='store_true')
    p.add_argument('--phase',choices=['prepare','native','qualify','smoke','recovery','train','pipeline'],default='prepare')
    p.add_argument('--arm',choices=['rgb_only','rgb_tracks'],default='rgb_tracks')
    p.add_argument('--replicate',type=int,choices=[0,1],default=0);args=p.parse_args()
    if not args.enable_track_training:raise SystemExit('Track training default off')
    r=a.install(True);old.worker=worker;assert not r.stopped()
    if args.phase=='prepare':prepare()
    elif args.phase=='native':native()
    elif args.phase=='qualify':old.qualify()
    elif args.phase=='smoke':r.train(args.arm,True)
    elif args.phase=='recovery':r.recovery_worker(args.arm,args.replicate)
    elif args.phase=='train':r.train(args.arm)
    else:old.pipeline()

if __name__=='__main__':main()
