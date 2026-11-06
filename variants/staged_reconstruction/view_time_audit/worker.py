"""Opt-in frozen model render audit. Never constructs a training objective."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import msvcrt
import numpy as np
from PIL import Image
import torch
from ..prepare_data import ROOT, DATA, read, sha
from ..joint_rgb.io import atomic_json
from .core import pairs, describe, angle_summary, pose_distance

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_view_time_audit'
REPORT=BASE/'v1'
RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_view_time_audit/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_view_time_audit/v1'

def stopped():
    return any(p.exists() for p in [ROOT/'stop.request',RUNS/'stop.request',RUNS.parent/'stop.request',RUNS.parent.parent/'stop.request'])

def tensor_hash(value):
    array=value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(str((array.shape,str(array.dtype))).encode()+array.tobytes()).hexdigest()

def state_digest(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):h.update(tensor_hash(x).encode())
        elif isinstance(x,np.ndarray):h.update(str((x.shape,str(x.dtype))).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k,v in sorted(x.items(),key=lambda kv:str(kv[0])):h.update(str(k).encode());visit(v)
        elif isinstance(x,(list,tuple)):
            h.update(str(type(x)).encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

def load_model(name,p):
    from ..j7_long_memory import runtime as adapter
    r=adapter.install(True)
    arm='rgb_control' if name=='J7' else name
    renderer,scene,opts,teacher,parent=r.load(arm)
    path=ROOT/p['models'][name]['path'];assert sha(path)==p['models'][name]['sha256']
    state=parent if name=='J7' else r.restore(path,scene,opts)
    r.assert_equivalent(scene.state_dict(),state['model'],0,0)
    for opt,original in zip(opts,state['optimizers']):r.assert_equivalent(opt.state_dict(),original,0,0)
    return adapter,r,renderer,scene,opts,state

def snapshot(r,scene,opts,state):
    return deepcopy(r.cpu_values(dict(full=r.snapshot(scene,opts,state['iteration'],state['camera_stack'],state['sample_trace']),
        flags=[p.requires_grad for p in scene.parameters()],training=[m.training for m in scene.modules()],
        gradients={k:None if p.grad is None else p.grad for k,p in scene.named_parameters()})))

def native(renderer,scene,cam,dynamic,gray=False,joint=False):
    from ..stage2.model import compose
    from ..train_static import PIPE
    bank=compose(scene.static,[dynamic]) if joint else dynamic
    color=torch.full((len(dynamic.get_xyz),3),.5,device='cuda') if gray else None
    return renderer(cam,bank,PIPE,scene.centers.new_zeros(3),0.,0.,0.,override_color=color)

def hashes(pkg):return {k:tensor_hash(pkg[k]) for k in ['render','alpha','depth']}

def save_png(path,value,receipts):
    path.parent.mkdir(parents=True,exist_ok=True)
    array=value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value
    if array.ndim==3 and array.shape[0] in [1,3]:array=array.transpose(1,2,0)
    array=np.squeeze(array) if array.shape[-1:]==(1,) else array
    Image.fromarray((np.clip(array,0,1)*255).round().astype(np.uint8)).save(path)
    receipts[path.relative_to(ROOT).as_posix()]=sha(path)

def verify_freeze():
    for receipt in [BASE/'preregistration.json',REPORT/'source_freeze.json']:
        for name,h in read(receipt)['files'].items():assert sha(ROOT/name)==h,name

def freeze():
    p=read(BASE/'protocol.json');files={}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if not name:continue
        path=Path(name).resolve()
        try:rel=path.relative_to(ROOT).as_posix()
        except ValueError:continue
        if path.is_file() and not rel.startswith(('envs/','tmp/','tmp_build/')):files[rel]=sha(path)
    inherited=read(ROOT/'experiments/staged_reconstruction/RubberDuck_J7_long_memory/v1/source_freeze.json')
    for rel,h in inherited.get('files',inherited).items():
        assert sha(ROOT/rel)==h,rel
        files[rel]=h
    for root in [ROOT/'variants/staged_reconstruction/view_time_audit',BASE]:
        for path in root.glob('*.py'):files[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in [BASE/'plan.md',BASE/'protocol.json',ROOT/'tests/test_view_time_audit.py',
                 ROOT/p['metadata'],ROOT/'experiments/staged_reconstruction/RubberDuck_J7_long_memory/v1/source_freeze.json']:
        files[path.relative_to(ROOT).as_posix()]=sha(path)
    for item in p['models'].values():files[item['path']]=sha(ROOT/item['path']);assert files[item['path']]==item['sha256']
    for frame in read(ROOT/p['metadata'])['frames']:
        path=DATA/(frame['file_path']+'.png')
        if not path.exists():path=ROOT/'data/PIDG/rubber_duck/train'/path.name
        files[path.relative_to(ROOT).as_posix()]=sha(path)
    atomic_json(REPORT/'source_freeze.json',dict(files=dict(sorted(files.items())),before_full_audit=True))
    verify_freeze()

def qualification():
    p=read(BASE/'protocol.json');meta=read(ROOT/p['metadata']);results={}
    from ..train_static import camera
    for name in p['models']:
        assert not stopped()
        adapter,r,renderer,scene,opts,state=load_model(name,p)
        before=snapshot(r,scene,opts,state)
        with adapter.memory.evidence_context(scene):
            frame=meta['frames'][24];cam=camera(frame,meta,p['width']);dynamic=scene.dynamic_at(frame['time'])
            direct=scene.render(renderer,cam,frame['time']);joint=native(renderer,scene,cam,dynamic,joint=True)
            assert all(torch.equal(direct[k],joint[k]) for k in ['render','alpha','depth'])
            regular=native(renderer,scene,cam,dynamic);gray=native(renderer,scene,cam,dynamic,gray=True)
            assert all(torch.equal(regular[k],gray[k]) for k in ['alpha','depth'])
            assert all(torch.isfinite(pkg[k]).all() for pkg in [direct,regular,gray] for k in ['render','alpha','depth'])
        r.assert_equivalent(before,snapshot(r,scene,opts,state),0,0)
        try:
            with adapter.memory.evidence_context(scene):
                next(scene.parameters()).requires_grad_(not next(scene.parameters()).requires_grad)
                torch.rand(3,device='cuda');np.random.rand(3)
                raise ValueError('intentional no-mutation exception fixture')
        except ValueError:pass
        r.assert_equivalent(before,snapshot(r,scene,opts,state),0,0)
        results[name]=dict(native_parity=True,gray_alpha_depth_exact=True,exception_restoration=True,
            full_state_unchanged=True,state_sha256=state_digest(before),render_hashes=hashes(direct))
        print('V0_QUALIFIED',name,flush=True)
        del before,scene,opts,state,renderer,dynamic,direct,joint,regular,gray
    atomic_json(REPORT/'qualification.json',dict(passed=True,models=results))
    freeze()

def run(replay=False):
    p=read(BASE/'protocol.json');verify_freeze();assert read(REPORT/'qualification.json')['passed']
    meta=read(ROOT/p['metadata']);frames=meta['frames'];selected=pairs(p)
    from ..train_static import camera
    all_results={}
    for name in p['models']:
        assert not stopped()
        adapter,r,renderer,scene,opts,state=load_model(name,p);before=snapshot(r,scene,opts,state)
        rows=[];receipts={};native_receipts={}
        reference=read(REPORT/(name+'.json')) if replay else None
        original_rows={(v['camera_index'],v['time_index']):v for v in reference['rows']} if replay else None
        with adapter.memory.evidence_context(scene):
            cameras={c:camera(frames[c],meta,p['width']) for c in p['all_camera_indices']}
            for n,(c,t) in enumerate(selected):
                if stopped():raise RuntimeError('User stop: read-only job stopped, no automatic resume')
                dynamic=scene.dynamic_at(frames[t]['time']);joint=native(renderer,scene,cameras[c],dynamic,joint=True)
                dyn=native(renderer,scene,cameras[c],dynamic)
                a=dyn['alpha'].squeeze().cpu().numpy();rgb=dyn['render'].permute(1,2,0).cpu().numpy();d=dyn['depth'].squeeze().cpu().numpy()
                row=dict(camera_index=c,time_index=t,time=frames[t]['time'],
                    camera_time_distance=pose_distance(frames[t]['transform_matrix'],frames[c]['transform_matrix']),
                    dynamic=describe(a,rgb,d,p,enabled=True),hashes=dict(joint=hashes(joint),dynamic=hashes(dyn)),observed_PSNR=None)
                if c==t:
                    target=r.frame_target(frames[c],p['width']);mse=float((target-joint['render']).square().mean())
                    row['observed_PSNR']=float(-10*np.log10(max(mse,1e-20)))
                if replay:assert row==original_rows[c,t],(name,c,t,'800 replay mismatch')
                else:
                    prefix=OUT/name/'cells'/f'c{c:03d}_t{t:03d}'
                    save_png(prefix.with_name(prefix.name+'_joint.png'),joint['render'],receipts)
                    save_png(prefix.with_name(prefix.name+'_dynamic.png'),dyn['render'],receipts)
                    save_png(prefix.with_name(prefix.name+'_alpha.png'),dyn['alpha'],receipts)
                    if c==t:save_png(prefix.with_name(prefix.name+'_observed.png'),target,receipts)
                if t in p['anchors'] and c in [t+o for o in p['offsets']]:
                    gray=native(renderer,scene,cameras[c],dynamic,gray=True)
                    assert all(torch.equal(gray[k],dyn[k]) for k in ['alpha','depth'])
                    cam1600=camera(frames[c],meta,p['native_export_width'])
                    bigjoint=native(renderer,scene,cam1600,dynamic,joint=True)
                    bigdyn=native(renderer,scene,cam1600,dynamic);biggray=native(renderer,scene,cam1600,dynamic,gray=True)
                    assert all(torch.equal(biggray[k],bigdyn[k]) for k in ['alpha','depth'])
                    item=dict(width800_gray=hashes(gray),joint=hashes(bigjoint),dynamic=hashes(bigdyn),gray=hashes(biggray))
                    key=f'c{c:03d}_t{t:03d}'
                    if replay:assert item==reference['native1600'][key],(name,key,'1600 replay mismatch')
                    else:
                        for label,pkg in [('joint',bigjoint),('dynamic',bigdyn),('gray',biggray)]:
                            save_png(OUT/name/'native1600'/f'{key}_{label}.png',pkg['render'],receipts)
                        save_png(OUT/name/'native1600'/f'{key}_alpha.png',bigdyn['alpha'],receipts)
                        save_png(OUT/name/'cells'/f'{key}_gray.png',gray['render'],receipts)
                    native_receipts[key]=item
                    del gray,bigjoint,bigdyn,biggray
                rows.append(row)
                if n==0 or (n+1)%25==0:
                    atomic_json(RUNS/'active.json',dict(status='independent_replay' if replay else 'rendering',model=name,completed=n+1,total=len(selected),pid=os.getpid()))
                    print('V0_RENDER',name,'replay' if replay else 'main',n+1,len(selected),flush=True)
                del joint,dyn,dynamic
        after=snapshot(r,scene,opts,state);r.assert_equivalent(before,after,0,0)
        result=dict(model=name,checkpoint=p['models'][name],pairs=len(rows),rows=rows,
            summary=angle_summary(rows,frames,p),native1600=native_receipts,
            state_sha256_before=state_digest(before),state_sha256_after=state_digest(after),state_unchanged=True)
        if replay:
            assert result['summary']==reference['summary'] and result['native1600']==reference['native1600']
            assert result['state_sha256_before']==reference['state_sha256_before']
            for path,h in reference['output_files'].items():assert sha(ROOT/path)==h,path
            all_results[name]=dict(passed=True,pairs=len(rows),native1600=len(native_receipts),all_metrics_hashes_and_outputs_exact=True)
        else:
            result['output_files']=receipts;atomic_json(REPORT/(name+'.json'),result)
        del scene,opts,state,before,after,cameras;torch.cuda.empty_cache()
    verify_freeze()
    if replay:atomic_json(REPORT/'independent_audit.json',dict(passed=True,models=all_results,process=os.getpid()))
    atomic_json(RUNS/'final_status.json',dict(status='independent_replay_complete_pending_visual_review' if replay else 'rendered_pending_replay',pending_review=True,training_updates=0))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enabled',action='store_true')
    parser.add_argument('--mode',choices=['qualify','freeze','run','replay'],default='run');args=parser.parse_args()
    if not args.enabled:print('V0_DISABLED');return
    torch.set_num_threads(4);REPORT.mkdir(parents=True,exist_ok=True);RUNS.mkdir(parents=True,exist_ok=True)
    assert not stopped()
    with (RUNS/'audit.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            if args.mode=='qualify':qualification()
            elif args.mode=='freeze':
                assert read(REPORT/'qualification.json')['passed']
                for name in read(BASE/'protocol.json')['models']:
                    loaded=load_model(name,read(BASE/'protocol.json'));del loaded
                freeze()
            else:run(args.mode=='replay')
        except Exception as error:
            atomic_json(RUNS/f'failure_{args.mode}_{time.time_ns()}.json',dict(error=repr(error),no_training=True,pid=os.getpid()))
            raise
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__':main()
