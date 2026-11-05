"""Explicitly opt-in J6 adapter; frozen J4 implementation is never edited."""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime,timezone
from variants.staged_reconstruction.i10_joint import runtime as adapter
from variants.staged_reconstruction.prepare_data import ROOT,DATA,read,sha
from experiments.staged_reconstruction.RubberDuck_joint_pilot import launch as original


def worker(phase,arm='joint_depth_trust',replicate=0):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=adapter.REPORT/f'{stamp}_{phase}_{arm}_{replicate}.log'
    path.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--enable-i10-joint',
        '--phase',phase,'--arm',arm,'--replicate',str(replicate)]
    with path.open('x',encoding='utf-8') as stream:
        code=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
    assert code==0,f'Preserve failed log: {path}'
    return {'log':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}


def prepare():
    import torch
    r=adapter.install(True)
    assert not r.stopped() and not (r.REPORT/'source_freeze.json').exists()
    assert read(r.REPORT/'unit_tests.json')['passed']
    torch.set_num_threads(4)
    _,scene,opts,_,parent=r.load('joint_depth_trust')
    assert len(scene.static.get_xyz)==4126 and len(scene.pools[0].get_xyz)==15507
    counters=lambda opt:{g['name']:sorted({int(opt['state'][key]['step']) for key in g['params'] if key in opt['state']}) for g in opt['param_groups']}
    r.atomic_json(r.REPORT/'parent_audit.json',{
        'static_points':4126,'dynamic_points':15507,'component_parents':read(r.BASE/'protocol.json')['parent_state_policy'],
        'static_Adam_steps':counters(opts[0].state_dict()),'dynamic_Adam_steps':counters(opts[1].state_dict()),
        'static_geometry_and_coordinate_frame_exact':True,'all_non_static_model_Adam_RNG_graphs_exact_I10':True,
        'static_and_complete_static_Adam_exact_J4':True,'legacy_conditional_color_masks':True,
        'old_I10_target_neutralization':False,'physics_frozen_RGB_only':True,
        'camera_stack':len(parent['camera_stack']),'formal_updates':0})
    files={}
    for receipt in [ROOT/'experiments/staged_reconstruction/RubberDuck_joint_pilot/v1/source_freeze.json',
                    ROOT/'experiments/staged_reconstruction/background_residual/source_freeze.json']:
        for name,h in read(receipt).items():
            assert sha(ROOT/name)==h,name
            assert name not in files or files[name]==h,name
            files[name]=h
    p=read(r.BASE/'protocol.json')
    paths=[ROOT/p[key] for key in ['parent','static_parent']]
    paths += [DATA/'transforms_train.json',r.BASE/'protocol.json',r.BASE/'plan.md',r.BASE/'preregistration.json',
        r.BASE/'preregister.py',r.BASE/'launch.py',ROOT/'tests/test_i10_joint.py']
    paths += [DATA/(f['file_path']+'.png') for f in read(DATA/'transforms_train.json')['frames']]
    paths += list((ROOT/'variants/staged_reconstruction/i10_joint').glob('*.py'))
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
    print('J6_COMPONENTS_AND_TRAIN_SOURCE_FROZEN',len(files),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-i10-joint',action='store_true')
    p.add_argument('--phase',choices=['prepare','native','qualify','smoke','recovery','train','pipeline'],default='prepare')
    p.add_argument('--arm',choices=['frozen_static','joint_depth_trust'],default='joint_depth_trust')
    p.add_argument('--replicate',type=int,choices=[0,1],default=0);args=p.parse_args()
    if not args.enable_i10_joint:raise SystemExit('J6 default off; pass --enable-i10-joint')
    r=adapter.install(True);original.worker=worker
    assert not r.stopped()
    if args.phase=='prepare':prepare()
    elif args.phase=='native':original.native()
    elif args.phase=='qualify':original.qualify()
    elif args.phase=='smoke':r.train(args.arm,True)
    elif args.phase=='recovery':r.recovery_worker(args.arm,args.replicate)
    elif args.phase=='train':r.train(args.arm)
    elif args.phase=='pipeline':original.pipeline()


if __name__=='__main__':main()
