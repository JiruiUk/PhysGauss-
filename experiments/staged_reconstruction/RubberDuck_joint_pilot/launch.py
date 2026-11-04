"""One opt-in real pilot, serial qualification/training; never a parameter sweep."""
import argparse
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from variants.staged_reconstruction.rubber_joint import runtime as r
from variants.staged_reconstruction.prepare_data import ROOT,DATA,read,sha


def worker(phase,arm='joint_depth_trust',replicate=0):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=r.REPORT/f'{stamp}_{phase}_{arm}_{replicate}.log';path.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--enable-rubber-joint','--phase',phase,
        '--arm',arm,'--replicate',str(replicate)]
    with path.open('x',encoding='utf-8') as stream:
        code=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
    assert code==0,f'Preserve failed log: {path}'
    return {'log':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}


def prepare():
    import torch
    assert not r.stopped() and not (r.REPORT/'source_freeze.json').exists()
    assert read(r.REPORT/'unit_tests.json')['passed']
    renderer,scene,opts,teacher,parent=r.load('joint_depth_trust')
    assert len(scene.static.get_xyz)==4126 and len(scene.pools[0].get_xyz)==15507
    r.atomic_json(r.REPORT/'parent_audit.json',{'static_points':4126,'dynamic_points':15507,
        'static_weights_exact_S1_and_Adam_step12000_inherited':True,'parent_dynamic_Adam_RNG_stack_graphs_inherited':True,
        'legacy_conditional_color_masks':True,'physics_frozen_reconstruction_only':True,
        'parent_camera_stack':len(parent['camera_stack']),'new_formal_updates':0})
    files=read(ROOT/'experiments/staged_reconstruction/S1_joint_revocation_audit/v1/preflight.json')['imported_sources']
    for name,h in files.items():assert sha(ROOT/name)==h,name
    p=read(r.BASE/'protocol.json')
    for name in ['parent','static_parent']:
        path=ROOT/p[name];files[path.relative_to(ROOT).as_posix()]=sha(path)
    meta=read(DATA/'transforms_train.json')
    paths=[DATA/'transforms_train.json',r.BASE/'protocol.json',r.BASE/'plan.md',r.BASE/'preregistration.json',
        ROOT/'tests/test_rubber_joint.py',ROOT/'variants/staged_reconstruction/joint_depth_trust/core.py',
        ROOT/'variants/staged_reconstruction/motion_integrity/prepare.py']
    paths += [DATA/(f['file_path']+'.png') for f in meta['frames']]
    paths += list((ROOT/'variants/staged_reconstruction/rubber_joint').glob('*.py'))
    paths += [r.BASE/'launch.py']
    for module in list(sys.modules.values()):
        source=getattr(module,'__file__',None)
        if not source:continue
        path=Path(source).resolve()
        if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'envs'):paths.append(path)
    for path in paths:
        assert path.is_file(),path
        files[path.relative_to(ROOT).as_posix()]=sha(path)
    assert not any('/evaluation_only/' in name for name in files)
    r.atomic_json(r.REPORT/'source_freeze.json',files);r.verify()
    print('J4_TRAIN_INPUT_SOURCE_FREEZE',len(files),flush=True)


def native():
    import torch
    from variants.staged_reconstruction.train_static import camera
    from variants.staged_reconstruction.rubber_joint.core import configure,objective,depth_term
    from variants.staged_reconstruction.joint_rgb.core import assert_equivalent
    torch.set_num_threads(4);r.verify()
    renderer,scene,opts,_,parent=r.load('frozen_static')
    assert configure(scene) is scene
    meta=read(DATA/'transforms_train.json');frame=meta['frames'][0]
    cam=camera(frame,meta,1600);target=r.frame_target(frame)
    old,package,_,_=objective(renderer,scene,None,cam,frame['time'],target)
    native={k:package[k].detach().clone() for k in ['render','alpha','depth']}
    old.backward();grads={n:None if p.grad is None else p.grad.detach().clone()
        for n,p in scene.named_parameters() if not n.startswith('static.')}
    scene.zero_grad(set_to_none=True);del package,old
    teacher=configure(scene,True)
    loss,package,_,self_loss=objective(renderer,scene,teacher,cam,frame['time'],target,True)
    assert self_loss.item()==0
    assert all(torch.equal(native[k],package[k]) for k in native)
    loss.backward()
    assert_equivalent(grads,{n:p.grad for n,p in scene.named_parameters() if not n.startswith('static.')},1e-6,1e-5)
    assert all(getattr(scene.static,n).grad is not None and getattr(scene.static,n).grad.isfinite().all()
        for n in ['_opacity','_features_dc','_features_rest'])
    scene.zero_grad(set_to_none=True);del loss,package,self_loss,native,grads
    with torch.no_grad():
        rows=torch.arange(len(scene.static._opacity),device='cuda')
        scene.static._opacity[:,0].add_(.03+.01*torch.cos(rows))
    isolated=depth_term(renderer,scene,teacher,cam);isolated.backward()
    assert scene.static._opacity.grad.isfinite().all() and scene.static._opacity.grad.abs().max()>0
    assert all(p.grad is None or torch.equal(p.grad,torch.zeros_like(p.grad))
        for n,p in scene.named_parameters() if n!='static._opacity')
    r.atomic_json(r.REPORT/'native_check.json',{'passed':True,'native_width':1600,
        'flags_off_original_scene_identity':True,'initial_RGB_alpha_D_exact':True,
        'dynamic_initial_RGB_gradients_original_tolerance':True,'isolated_self_depth_only_opacity':True,
        'new_formal_updates':0,'J3_native_finite_difference_retained':True})
    r.verify();print('J4_NATIVE_PASS',flush=True)


def qualify():
    import torch
    from variants.staged_reconstruction.joint_rgb.core import assert_equivalent
    assert read(r.REPORT/'native_check.json')['passed'] and not (r.REPORT/'recovery.json').exists()
    rows=[]
    with r.lock('qualification.lock'):
        for arm in read(r.BASE/'protocol.json')['arms']:
            assert not r.stopped()
            logs=[worker('smoke',arm)]+[worker('recovery',arm,i) for i in [0,1]]
            directory=r.RUNS/'qualification'/arm
            for kind in ['restored','replay']:
                a,b=[torch.load(directory/f'{kind}{i}.pt',map_location='cpu',weights_only=False) for i in [0,1]]
                assert_equivalent(a,b,0,0)
            rows.append({'arm':arm,'passed':True,'logs':logs,'model_two_Adam_bindings_RNG_sampler_graphs_exact':True,
                'native_render_exact_gradients_original_tolerance':True,'same_native_gradient_Adam_update_exact':True})
    r.verify();r.atomic_json(r.REPORT/'recovery.json',{'passed':True,'rows':rows,
        'human_approved_acceptance_inherited':True,'old_full40_failures_preserved':True})


def pipeline():
    import os
    assert read(r.REPORT/'recovery.json')['passed'] and not r.stopped()
    with r.lock('pipeline.lock'):
        state={'status':'training','pid':os.getpid(),'completed':[]}
        r.atomic_json(r.RUNS/'pipeline.json',state)
        try:
            for arm in read(r.BASE/'protocol.json')['arms']:
                receipt=r.RUNS/'formal'/arm/'checkpoint.json'
                if receipt.exists():
                    assert read(receipt)['iteration']==2000,'Do not resume a stopped/partial run automatically'
                else:state['completed'].append(worker('train',arm))
                if r.stopped():
                    state['status']='saved_stopped';r.atomic_json(r.RUNS/'pipeline.json',state);return
            state['status']='trained_pending_evaluation';r.atomic_json(r.RUNS/'pipeline.json',state)
        except Exception as error:
            state.update(status='failed',error=repr(error));r.atomic_json(r.RUNS/'pipeline.json',state);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-rubber-joint',action='store_true')
    p.add_argument('--phase',choices=['prepare','native','qualify','smoke','recovery','train','pipeline'],default='prepare')
    p.add_argument('--arm',choices=['frozen_static','joint_depth_trust'],default='joint_depth_trust')
    p.add_argument('--replicate',type=int,choices=[0,1],default=0);args=p.parse_args()
    if not args.enable_rubber_joint:raise SystemExit('J4 default off; pass --enable-rubber-joint')
    assert not r.stopped()
    if args.phase=='prepare':prepare()
    elif args.phase=='native':native()
    elif args.phase=='qualify':qualify()
    elif args.phase=='smoke':r.train(args.arm,True)
    elif args.phase=='recovery':r.recovery_worker(args.arm,args.replicate)
    elif args.phase=='train':r.train(args.arm)
    elif args.phase=='pipeline':pipeline()

if __name__=='__main__':main()
