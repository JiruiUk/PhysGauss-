"""Explicit S2 RGB training. Complete-state resume; S1 tensors never optimized."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ..prepare_data import ROOT, REPORT, RUNS, DATA, read, write, sha
from ..train_static import setup, camera
from ..model import observation_masks
from .model import ObjectScene, compose
from ..train_static import PIPE


def image(frame,width):
    path=DATA/(frame['file_path']+'.png')
    if not path.exists():path=ROOT/'data/PIDG/rubber_duck/train'/path.name
    a=np.asarray(Image.open(path).convert('RGBA').resize((width,width*9//16),Image.Resampling.BILINEAR),dtype=np.float32)/255
    return a


def checkpoint(directory,scene,opt,step,stack,protocol_hash):
    tmp=directory/'checkpoint.tmp';path=directory/'checkpoint.pt'
    torch.save({'model':scene.state_dict(),'optimizer':opt.state_dict(),'iteration':step,'camera_stack':stack,
        'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
        'rng_cuda':torch.cuda.get_rng_state_all(),'protocol_sha256':protocol_hash,'stage':'S2',
        'static_checkpoint_sha256':read(REPORT/'S2_protocol.json')['static_checkpoint_sha256'],
        'physical_ID_after_ambiguity':'unknown'},tmp)
    os.replace(tmp,path);write(directory/'checkpoint.json',{'iteration':step,'sha256':sha(path),'full_optimizer_RNG_sampler':True})


def restore(directory,scene,opt,protocol_hash):
    path=directory/'checkpoint.pt'
    if not path.exists():return 0,[]
    assert sha(path)==read(directory/'checkpoint.json')['sha256']
    s=torch.load(path,map_location='cpu',weights_only=False);assert s['protocol_sha256']==protocol_hash
    scene.load_state_dict(s['model']);opt.load_state_dict(s['optimizer'])
    random.setstate(s['rng_python']);np.random.set_state(s['rng_numpy']);torch.set_rng_state(s['rng_torch']);torch.cuda.set_rng_state_all(s['rng_cuda'])
    print('OBJECT_FULL_CHECKPOINT_RESTORED',s['iteration'],flush=True)
    return s['iteration'],s['camera_stack']


def objective(pred,gt):
    from utils.loss_utils import ssim
    return .8*(pred-gt).abs().mean()+.2*(1-ssim(pred,gt))


@torch.no_grad()
def evaluate(renderer,scene,step,split='val',width=1600,save_images=False):
    meta=read(ROOT/f'data/derived/kinematic_resplit/transforms_{split}.json');rows=[]
    out=ROOT/'output/staged_reconstruction/S2_objects';out.mkdir(parents=True,exist_ok=True)
    for frame in meta['frames']:
        rgba=image(frame,width);gt=torch.tensor(rgba[...,:3]*rgba[...,3:4],device='cuda').permute(2,0,1)
        cam=camera(frame,meta,width);pkg=scene.render(renderer,cam,frame['time']);pred=pkg['render'].clamp(0,1)
        static,bg=observation_masks(rgba,max(2,width//100))
        r,g,b=rgba[...,:3].transpose(2,0,1)
        dynamic=(rgba[...,3]>.95)&(r>.25)&(g>.18)&(b<.8*np.minimum(r,g))
        parts=[scene.object_at(k,frame['time']) for k in range(len(scene.objects))]
        dynamic_pkg=renderer(cam,compose(parts[0],parts[1:]),PIPE,scene.centers.new_zeros(3),0.,0.,0.)
        row={'frame':int(Path(frame['file_path']).stem),'PSNR':float(-10*torch.log10((pred-gt).square().mean().clamp_min(1e-12)))}
        for label,mask in [('dynamic',dynamic),('static',static)]:
            m=torch.tensor(mask,device='cuda')
            row[label+'_PSNR']=float(-10*torch.log10((pred[:,m]-gt[:,m]).square().mean().clamp_min(1e-12))) if m.any() else None
            row[label+'_alpha']=float(pkg['alpha'].squeeze()[m].mean()) if m.any() else None
        row['composite_dynamic_region_alpha']=row['dynamic_alpha']
        dm=torch.tensor(dynamic,device='cuda')
        row['dynamic_alpha']=float(dynamic_pkg['alpha'].squeeze()[dm].mean()) if dm.any() else None
        row['dynamic_leakage_on_static']=float(dynamic_pkg['alpha'].squeeze()[torch.tensor(static,device='cuda')].mean())
        row['background_alpha']=float(pkg['alpha'].squeeze()[torch.tensor(bg,device='cuda')].mean());rows.append(row)
        if save_images:
            Image.fromarray((pred.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(out/f'{split}_{row["frame"]:04d}.png')
    metrics={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0] if k!='frame'}
    result={'stage':'S2','iteration':step,'split':split,'width':width,'metrics':metrics,'frames':rows,
        'masks':'Conditional appearance masks, not independent physical ID or dense geometry GT'}
    write(REPORT/f'S2_{split}_{step:05d}_{width}.json',result)
    print('OBJECT_EVALUATION',step,split,width,metrics,flush=True);return result


def preflight(renderer,scene,opt,directory,protocol_hash):
    meta=read(DATA/'transforms_train.json');frame=meta['frames'][0];width=800
    target=image(frame,width);gt=torch.tensor(target[...,:3]*target[...,3:4],device='cuda').permute(2,0,1);cam=camera(frame,meta,width)
    frozen={n:p.detach().clone() for n,p in scene.static.state_dict().items()};ids=scene.object_ids.clone()
    gradients=[]
    for _ in range(3):
        opt.zero_grad(set_to_none=True);loss=objective(scene.render(renderer,cam,frame['time'])['render'],gt);loss.backward()
        groups={name:sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) for name,module in [('objects',scene.objects),('grids',scene.grids),('deform',scene.networks)]}
        assert all(math.isfinite(v) for v in groups.values())
        opt.step();scene.constrain();gradients.append(groups)
    assert all(v>0 for v in gradients[-1].values()),gradients
    assert all(p.grad is None and not p.requires_grad for p in scene.static.parameters())
    assert all(torch.equal(v,scene.static.state_dict()[n]) for n,v in frozen.items())
    assert torch.equal(ids,scene.object_ids)
    checkpoint(directory,scene,opt,3,[7,2],protocol_hash)
    expected={n:v.detach().clone() for n,v in scene.state_dict().items()}
    random_values=[random.random(),np.random.random(),float(torch.rand((),device='cuda'))]
    step,stack=restore(directory,scene,opt,protocol_hash);assert step==3 and stack==[7,2]
    assert random_values==[random.random(),np.random.random(),float(torch.rand((),device='cuda'))]
    assert all(torch.equal(v,scene.state_dict()[n]) for n,v in expected.items())
    # Native forward/backward must work before formal training.
    native=image(frame,1600);native_gt=torch.tensor(native[...,:3]*native[...,3:4],device='cuda').permute(2,0,1)
    opt.zero_grad(set_to_none=True);native_loss=objective(scene.render(renderer,camera(frame,meta,1600),frame['time'])['render'],native_gt);native_loss.backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
    from ..freeze import verify
    verify()
    write(REPORT/'S2_preflight.json',{'passed':True,'protocol_sha256':protocol_hash,'gradients':gradients,
        'static_tensors_exactly_unchanged':True,'static_optimizer_parameters':0,'immutable_object_membership':True,
        'optimizer_RNG_sampler_restored':True,'native_resolution_backward_finite':True,'CMR_contact_off':True,
        'dynamic_points':len(ids),'static_points':len(scene.static.get_xyz),'S1_source_freeze_passed':True})


def run(args):
    protocol=read(REPORT/'S2_protocol.json');protocol_hash=sha(REPORT/'S2_protocol.json')
    torch.set_num_threads(4);random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
    renderer=setup();scene=ObjectScene(protocol);opt=scene.optimizer()
    name='_S2_preflight' if args.preflight_only else '_S2_smoke' if args.smoke else 'S2_objects'
    directory=RUNS/name;directory.mkdir(parents=True,exist_ok=True)
    if args.preflight_only:preflight(renderer,scene,opt,directory,protocol_hash);return
    pre=read(REPORT/'S2_preflight.json');assert pre['passed'] and pre['protocol_sha256']==protocol_hash
    from ..freeze import verify
    verify()
    if not args.smoke:
        from .freeze import verify as verify_objects
        verify_objects()
    start,stack=restore(directory,scene,opt,protocol_hash)
    if args.evaluate_only:
        evaluate(renderer,scene,start,'val',1600,True);evaluate(renderer,scene,start,'test',1600,True);return
    static={n:p.detach().clone() for n,p in scene.static.state_dict().items()};ids=scene.object_ids.clone()
    meta=read(DATA/'transforms_train.json');cache={};cameras={};width=None;began=time.monotonic();last_completed=start
    formal=not args.smoke
    if formal:write(RUNS/'state.json',{'stage':'S2','status':'training','pid':os.getpid(),'iteration':start})
    try:
        for step in range(start+1,protocol['stage_updates']+1):
            next_width=800 if step<=12000 else 1600
            if next_width!=width:cache.clear();cameras.clear();width=next_width
            if not stack:stack=random.sample(range(len(meta['frames'])),len(meta['frames']))
            index=stack.pop();frame=meta['frames'][index]
            if index not in cache:
                rgba=image(frame,width);cache[index]=torch.tensor(rgba[...,:3]*rgba[...,3:4]).permute(2,0,1)
                cameras[index]=camera(frame,meta,width)
            gt=cache[index].cuda();opt.zero_grad(set_to_none=True)
            loss=objective(scene.render(renderer,cameras[index],frame['time'])['render'],gt)
            assert bool(torch.isfinite(loss));loss.backward()
            assert all(p.grad is None or bool(p.grad.isfinite().all()) for p in scene.parameters())
            opt.step();scene.constrain()
            last_completed=step
            if step%20==0:
                with (directory/'losses.jsonl').open('a') as f:f.write(json.dumps({'iteration':step,'frame':frame['file_path'],'loss':float(loss),'width':width})+'\n')
            if step%200==0:
                assert all(torch.equal(v,scene.static.state_dict()[n]) for n,v in static.items()) and torch.equal(ids,scene.object_ids)
                if formal:write(RUNS/'state.json',{'stage':'S2','status':'training','pid':os.getpid(),'iteration':step,'loss':float(loss),'width':width})
                print('OBJECT_TRAIN',step,'loss',float(loss),'elapsed',round(time.monotonic()-began,1),flush=True)
            if step%protocol['checkpoint_interval']==0:checkpoint(directory,scene,opt,step,stack,protocol_hash)
            if formal and step%protocol['validation_interval']==0:evaluate(renderer,scene,step,'val',800)
            if (RUNS/'stop.request').exists() or args.stop_after and step>=args.stop_after:
                checkpoint(directory,scene,opt,step,stack,protocol_hash)
                if formal:write(RUNS/'state.json',{'stage':'S2','status':'saved_stopped','iteration':step})
                return
        checkpoint(directory,scene,opt,protocol['stage_updates'],stack,protocol_hash)
        evaluate(renderer,scene,protocol['stage_updates'],'val',1600,True);evaluate(renderer,scene,protocol['stage_updates'],'test',1600,True)
        verify()
        if formal:write(RUNS/'state.json',{'stage':'S2','status':'evaluated_pending_review','iteration':protocol['stage_updates']})
    except Exception as error:
        checkpoint(directory,scene,opt,last_completed,stack,protocol_hash)
        if formal:write(RUNS/'state.json',{'stage':'S2','status':'failed','iteration':last_completed,'attempted_iteration':step,'error':str(error)})
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-object-stage',action='store_true');p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--evaluate-only',action='store_true');args=p.parse_args()
    if not args.enable_object_stage:raise SystemExit('S2 is opt-in; pass --enable-object-stage')
    # Native Windows process lock: crash releases the byte lock automatically.
    import msvcrt
    RUNS.mkdir(parents=True,exist_ok=True)
    lock=open(RUNS/('_S2_smoke.lock' if args.smoke or args.preflight_only else 'S2_training.lock'),'a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('S2 already running; did not start a duplicate')
    try:run(args)
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
