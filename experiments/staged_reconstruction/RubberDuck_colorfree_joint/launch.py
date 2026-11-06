"""One matched static-library replacement, fully opt-in and serialized."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import subprocess
import sys
from variants.staged_reconstruction.colorfree_joint import runtime as a
from variants.staged_reconstruction.prepare_data import ROOT, DATA, read, sha
from experiments.staged_reconstruction.RubberDuck_joint_pilot import launch as old


def worker(phase,arm='colorfree_static',replicate=0):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=a.REPORT/f'{stamp}_{phase}_{arm}_{replicate}.log';path.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--enable-colorfree-joint','--phase',phase,'--arm',arm,'--replicate',str(replicate)]
    with path.open('x',encoding='utf-8') as stream:
        code=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW).returncode
    assert code==0,f'Preserved failed log: {path}'
    return dict(log=path.relative_to(ROOT).as_posix(),sha256=sha(path))


def prepare():
    import torch
    r=a.install(True);assert not r.stopped() and not (a.REPORT/'source_freeze.json').exists()
    assert read(a.REPORT/'unit_tests.json')['passed'];torch.set_num_threads(4)
    rows={}
    for arm in read(a.BASE/'protocol.json')['arms']:
        _,scene,opts,teacher,parent=r.load(arm)
        rows[arm]=dict(static_points=len(scene.static.get_xyz),dynamic_points=len(scene.pools[0].get_xyz),
            non_static_model_and_complete_Adam_exact_J6=True,static_and_complete_Adam_exact_source=True,
            self_teacher_exact_own_initial_static=True,static_geometry_frozen=True,physics_frozen=True)
        r.assert_equivalent(scene.static.state_dict(),teacher.state_dict(),0,0)
        del scene,opts,teacher,parent
    files=read(ROOT/'experiments/staged_reconstruction/RubberDuck_I10_joint/v1/source_freeze.json').copy()
    for receipt in ['experiments/staged_reconstruction/S1_clean_visibility/v1/source_freeze.json']:
        for name,h in read(ROOT/receipt).items():
            assert name not in files or files[name]==h,name
            files[name]=h
    for name,h in files.items():assert sha(ROOT/name)==h,name
    paths=list((ROOT/'variants/staged_reconstruction/colorfree_joint').glob('*.py'))+list(a.BASE.glob('*.py'))
    paths += [a.BASE/'protocol.json',a.BASE/'plan.md',a.BASE/'preregistration.json',ROOT/'tests/test_colorfree_joint.py']
    p=read(a.BASE/'protocol.json');paths += [ROOT/p[k] for k in ['parent','static_parent','static_initialization']]
    paths += [DATA/'transforms_train.json']+[DATA/(f['file_path']+'.png') for f in read(DATA/'transforms_train.json')['frames']]
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'envs'):paths.append(path)
    for path in paths:files[path.relative_to(ROOT).as_posix()]=sha(path)
    r.atomic_json(a.REPORT/'parent_audit.json',dict(passed=True,arms=rows))
    r.atomic_json(a.REPORT/'source_freeze.json',files);r.verify()
    print('J7_SOURCES_FROZEN',len(files),flush=True)


def native():
    import torch
    from PIL import Image
    from variants.staged_reconstruction.train_static import camera,render_static,PIPE
    from variants.staged_reconstruction.stage2.model import compose
    from variants.staged_reconstruction.rubber_joint.core import objective,depth_term
    from variants.staged_reconstruction.colorfree_joint.core import replace_static
    from experiments.staged_reconstruction.RubberDuck_joint_pilot.evaluate import rgb
    r=a.install(True);r.verify();torch.set_num_threads(4);a.OUT.mkdir(parents=True,exist_ok=True)
    rows={};meta=read(DATA/'transforms_train.json');frame=meta['frames'][0];cam=camera(frame,meta,1600)
    for arm in read(a.BASE/'protocol.json')['arms']:
        renderer,scene,opts,teacher,parent=r.load(arm)
        before=deepcopy(r.cpu_values(r.snapshot(scene,opts,0,parent['camera_stack'],[])))
        assert replace_static(scene) is scene
        with torch.no_grad():
            package=scene.render(renderer,cam,frame['time'])
            direct=renderer(cam,compose(scene.static,[scene.dynamic_at(frame['time'])]),PIPE,scene.centers.new_zeros(3),0.,0.,0.)
            assert all(torch.equal(package[k],direct[k]) for k in ['render','alpha','depth'])
        loss,package,_,self_loss=objective(renderer,scene,teacher,cam,frame['time'],r.frame_target(frame),True)
        assert self_loss.item()==0;loss.backward()
        for name in ['static._opacity','static._features_dc','static._features_rest','pools.0._xyz']:
            g=dict(scene.named_parameters())[name].grad
            assert g is not None and g.isfinite().all() and g.abs().sum()>0,name
        assert all(p.grad is None for n,p in scene.named_parameters() if n.startswith('physics.') or n in ['static._xyz','static._scaling','static._rotation'])
        r.assert_equivalent(before,r.snapshot(scene,opts,0,parent['camera_stack'],[]),0,0)
        scene.zero_grad(set_to_none=True)
        with torch.no_grad():scene.static._opacity[:,0].add_(.03+.01*torch.cos(torch.arange(len(scene.static._opacity),device='cuda')))
        isolated=depth_term(renderer,scene,teacher,cam);isolated.backward()
        assert scene.static._opacity.grad.isfinite().all() and scene.static._opacity.grad.abs().max()>0
        assert all(p.grad is None or torch.equal(p.grad,torch.zeros_like(p.grad)) for n,p in scene.named_parameters() if n!='static._opacity')
        rows[arm]=dict(passed=True,native_RGB_alpha_D_exact=True,own_teacher_initial_depth_zero=True,
                      intended_finite_nonzero_gradients=True,isolated_depth_opacity_only=True,full_state_no_optimizer_mutation=True)
        renderer,scene,opts,teacher,parent=r.load(arm)
        byid={int(Path(f['file_path']).stem):f for f in meta['frames']}
        with torch.no_grad():
            for cid in [86,188]:
                c=camera(byid[cid],meta,1600)
                Image.fromarray(rgb(render_static(renderer,c,scene.static))).save(a.OUT/f'initial_c{cid:03d}_{arm}_staticRGB_native.png')
                for number in [2,100,200]:
                    Image.fromarray(rgb(scene.render(renderer,c,(number-1)/199))).save(a.OUT/f'initial_c{cid:03d}_t{number:03d}_{arm}_jointRGB_native.png')
        del scene,opts,teacher,parent
    r.atomic_json(a.REPORT/'native_check.json',dict(passed=True,arms=rows,formal_updates=0));r.verify()
    print('J7_NATIVE_FLAGS_OFF_COMPOSITION_GRADIENT_PASS',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-colorfree-joint',action='store_true')
    p.add_argument('--phase',choices=['prepare','native','qualify','smoke','recovery','train','pipeline'],default='prepare')
    p.add_argument('--arm',choices=['old_static_control','colorfree_static'],default='colorfree_static')
    p.add_argument('--replicate',type=int,choices=[0,1],default=0);args=p.parse_args()
    if not args.enable_colorfree_joint:raise SystemExit('J7 is default off; explicit --enable-colorfree-joint required')
    r=a.install(True);assert not r.stopped();old.worker=worker
    if args.phase=='prepare':prepare()
    elif args.phase=='native':native()
    elif args.phase=='qualify':old.qualify()
    elif args.phase=='smoke':r.train(args.arm,True)
    elif args.phase=='recovery':r.recovery_worker(args.arm,args.replicate)
    elif args.phase=='train':r.train(args.arm)
    elif args.phase=='pipeline':old.pipeline()


if __name__=='__main__':main()
