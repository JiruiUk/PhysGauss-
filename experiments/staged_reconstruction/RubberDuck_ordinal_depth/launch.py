"""Single fixed pilot: input preparation, native qualification and two arms."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import numpy as np
from variants.staged_reconstruction.i10_ordinal_depth import runtime as a
from variants.staged_reconstruction.i10_ordinal_depth.core import make_pairs,ranking
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,sha
from experiments.staged_reconstruction.RubberDuck_joint_pilot import launch as old


def worker(phase,arm='ordinal_depth',replicate=0):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');path=a.REPORT/f'{stamp}_{phase}_{arm}_{replicate}.log'
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--enable-ordinal-depth','--phase',phase,'--arm',arm,'--replicate',str(replicate)]
    with path.open('x',encoding='utf-8') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,f'Preserve failed log: {path}'
    return dict(log=path.relative_to(ROOT).as_posix(),sha256=sha(path))


def prepare():
    import torch
    from variants.staged_reconstruction.state_recovery.runtime import graph_state
    r=a.install(True);assert not r.stopped() and not (a.REPORT/'source_freeze.json').exists()
    assert read(a.REPORT/'unit_tests.json')['passed'];torch.set_num_threads(4)
    p=read(a.BASE/'protocol.json');e=read(ROOT/p['evidence']);raw=[]
    for i in range(120):
        with np.load(ROOT/f'runs/pidg_staged_reconstruction/RubberDuck_pseudo_depth/v1/raw/{i:03d}.npz') as f:raw.append(f['relative'].copy())
    bank=dict(fit=make_pairs(e,raw,p),held=make_pairs(e,raw,p,True),absolute_depth_supervision=False,old_global_gates_failed=True)
    r.atomic_json(a.REPORT/'pair_bank.json',bank)
    renderer,scene,opts,_,parent=r.load('ordinal_depth')
    capture=lambda:r.cpu_values(dict(model=scene.state_dict(),optimizers=[o.state_dict() for o in opts],rng=a.rng_state(),camera=parent['camera_stack'],graphs=graph_state(scene)))
    before=deepcopy(capture());summary=a.refresh(renderer,scene,True);r.assert_equivalent(before,capture(),0,0)
    r.atomic_json(a.REPORT/'initial_health.json',dict(controller=a.STATE,summary=summary,full_state_unchanged=True))
    passed=summary['active']>=p['minimum_initial_pairs_for_local_pilot']
    r.atomic_json(a.REPORT/'local_support.json',dict(passed=passed,summary=summary,
        fit_pairs=len(bank['fit']['pairs']),held_pairs=len(bank['held']['pairs']),old_global_gates_not_revised=True,
        user_authorized_local_exploration=True,optimizer_updates=0))
    files=read(ROOT/'experiments/staged_reconstruction/RubberDuck_revocable_springs/v1/source_freeze.json').copy()
    for name,h in files.items():assert sha(ROOT/name)==h,name
    inherited=read(ROOT/p['evidence_freeze'])['files']
    for name,h in inherited.items():assert sha(ROOT/name)==h,name;files[name]=h
    paths=list((ROOT/'variants/staged_reconstruction/i10_ordinal_depth').glob('*.py'))+list(a.BASE.glob('*.py'))
    paths += [a.BASE/'plan.md',a.BASE/'protocol.json',a.BASE/'preregistration.json',ROOT/'tests/test_i10_ordinal_depth.py',
              a.REPORT/'pair_bank.json',a.REPORT/'initial_health.json',ROOT/p['evidence']]
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'envs'):paths.append(path)
    for path in paths:files[path.relative_to(ROOT).as_posix()]=sha(path)
    r.atomic_json(a.REPORT/'source_freeze.json',files);r.verify()
    print('ORDINAL_INITIAL_SUPPORT',passed,summary,'fit',len(bank['fit']['pairs']),'held',len(bank['held']['pairs']),flush=True)
    assert passed,'No native-supported ordinal pair; do not invent targets or relax gates'


def native():
    import torch
    from variants.staged_reconstruction.train_static import camera
    from variants.staged_reconstruction.rubber_joint.core import objective as baseline
    r=a.install(True);r.verify();torch.set_num_threads(4);renderer,scene,opts,_,parent=r.load('ordinal_depth')
    before=deepcopy(r.cpu_values(r.snapshot(scene,opts,0,parent['camera_stack'],[])))
    assert a.measure(renderer,scene)=={k:s['measurement'] for k,s in a.STATE['pairs'].items()}
    try:
        with a.evidence_context(scene):
            torch.rand(4,device='cuda');next(scene.parameters()).requires_grad_(False);raise ValueError('expected fixture')
    except ValueError:pass
    r.assert_equivalent(before,r.snapshot(scene,opts,0,parent['camera_stack'],[]),0,0)
    index=next(q['frame'] for q in a.BANK['fit']['pairs'] if a.STATE['pairs'][q['id']]['status']=='active')
    frame=a.META['frames'][index];cam=camera(frame,a.META,1600);target=r.frame_target(frame)
    loss,pkg,_,_=baseline(renderer,scene,None,cam,frame['time'],target)
    disabled,_=ranking(loss,enabled=False);assert disabled is loss
    images={k:pkg[k].detach().clone() for k in ['render','alpha','depth']};disabled.backward()
    gradients={n:None if p.grad is None else p.grad.detach().clone() for n,p in scene.named_parameters()}
    scene.zero_grad(set_to_none=True);del loss,pkg,disabled
    loss,pkg,_,_=baseline(renderer,scene,None,cam,frame['time'],target);loss.backward()
    assert all(torch.equal(images[k],pkg[k]) for k in images)
    r.assert_equivalent(gradients,{n:p.grad for n,p in scene.named_parameters()},1e-6,1e-5)
    scene.zero_grad(set_to_none=True);del loss,pkg,gradients,images
    natural,counts=a.ordinal(renderer,scene,index);natural_value=float(natural.detach());del natural
    fixture,fc=a.ordinal(renderer,scene,index,force_fixture=True);assert fixture>0;fixture.backward();norms={}
    for prefix in ['pools.0._xyz','grids.','networks.']:
        values=[p.grad for n,p in scene.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert values and all(torch.isfinite(v).all() for v in values)
        norms[prefix]=sum(float(v.abs().sum()) for v in values);assert norms[prefix]>0
    assert all(p.grad is None for n,p in scene.named_parameters() if n.startswith(('static.','physics.')))
    scene.zero_grad(set_to_none=True);del fixture
    r.assert_equivalent(before,r.snapshot(scene,opts,0,parent['camera_stack'],[]),0,0)
    r.atomic_json(a.REPORT/'native_check.json',dict(passed=True,width=1600,ordinal_width=400,
        flags_off_RGB_alpha_D_exact=True,flags_off_gradients_original_tolerance=True,
        independent_native_measurement_exact=True,no_mutation_exception_flags_rng=True,
        actual_ordinal_loss=natural_value,actual_counts=counts,forced_violation_gradient_fixture_only=True,
        isolated_native_depth_gradient_norms=norms,static_physics_frozen=True,optimizer_updates=0))
    r.verify();print('ORDINAL_NATIVE_PASS',natural_value,counts,norms,flush=True)


def pipeline():
    r=a.install(True);assert read(a.REPORT/'recovery.json')['passed'] and not r.stopped()
    with r.lock('pipeline.lock'):
        state=dict(status='training',completed=[]);r.atomic_json(a.RUNS/'pipeline.json',state)
        try:
            for arm in read(a.BASE/'protocol.json')['arms']:
                path=a.RUNS/'formal'/arm/'checkpoint.json'
                if path.exists():assert read(path)['iteration']==600,'Never resume partial/paused work automatically'
                else:state['completed'].append(worker('train',arm))
                if r.stopped():state['status']='saved_stopped';r.atomic_json(a.RUNS/'pipeline.json',state);return
            state['status']='trained_pending_evaluation';r.atomic_json(a.RUNS/'pipeline.json',state)
        except BaseException as error:
            state.update(status='failed',error=repr(error));r.atomic_json(a.RUNS/'pipeline.json',state);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-ordinal-depth',action='store_true')
    p.add_argument('--phase',choices=['prepare','native','qualify','smoke','recovery','train','pipeline'],default='prepare')
    p.add_argument('--arm',choices=['rgb_control','ordinal_depth'],default='ordinal_depth');p.add_argument('--replicate',type=int,choices=[0,1],default=0);args=p.parse_args()
    if not args.enable_ordinal_depth:raise SystemExit('Ordinal pilot default off')
    r=a.install(True);assert not r.stopped();old.worker=worker
    if args.phase=='prepare':prepare()
    elif args.phase=='native':native()
    elif args.phase=='qualify':old.qualify()
    elif args.phase=='smoke':r.train(args.arm,True)
    elif args.phase=='recovery':r.recovery_worker(args.arm,args.replicate)
    elif args.phase=='train':r.train(args.arm)
    else:pipeline()


if __name__=='__main__':main()
