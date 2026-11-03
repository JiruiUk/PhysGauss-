"""Explicit opt-in S1, complete-state recovery, native metrics and evidence gates."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from .prepare_data import ROOT, REPORT, RUNS, DATA, read, write, sha, image_data
from .model import masked_objective
from .static_gaussian import StaticGaussian

REPO=ROOT/'variants/color_repair/code'
PIPE=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False)


def setup():
    # Reuse the existing renderer verbatim; do not import the R0 warp policy.
    sys.path.insert(0,str(REPO))
    os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
    os.environ.pop('PIDG_METRIC_CONFIG',None)
    from gaussian_renderer import render
    assert Path(sys.modules['gaussian_renderer'].__file__).resolve()==REPO/'gaussian_renderer/__init__.py'
    return render


def camera(frame,meta,width):
    from utils.graphics_utils import getProjectionMatrix
    from output.short_clip_diagnostic.camera_math import w2c
    fovx=meta['camera_angle_x'];fovy=2*math.atan(math.tan(fovx/2)*9/16)
    view=torch.tensor(w2c(frame['transform_matrix'],'metadata_cv').T,dtype=torch.float32,device='cuda')
    projection=getProjectionMatrix(znear=.01,zfar=100.,fovX=fovx,fovY=fovy).T.cuda()
    return SimpleNamespace(FoVx=fovx,FoVy=fovy,image_width=width,image_height=width*9//16,
        world_view_transform=view,full_proj_transform=view@projection,camera_center=view.inverse()[3,:3])


def render_static(render,cam,g):
    return render(cam,g,PIPE,torch.zeros(3,device='cuda'),0.,0.,0.)


def optimizer_for(g):
    rates={'_xyz':.00008,'_features_dc':.0025,'_features_rest':.000125,
           '_opacity':.01,'_scaling':.002,'_rotation':.001}
    return torch.optim.Adam([{'params':[getattr(g,n)],'lr':rate,'name':n} for n,rate in rates.items()],eps=1e-15)


def save(directory,g,opt,step,stack,protocol_hash):
    path=directory/'checkpoint.pt';temp=directory/'checkpoint.tmp'
    torch.save({'model':g.state_dict(),'optimizer':opt.state_dict(),'iteration':step,'camera_stack':stack,
        'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
        'rng_cuda':torch.cuda.get_rng_state_all(),'protocol_sha256':protocol_hash,
        'static_only':True,'dynamic_parameters':0},temp)
    os.replace(temp,path)
    write(directory/'checkpoint.json',{'iteration':step,'sha256':sha(path),'optimizer_RNG_sampler_saved':True})


def restore(directory,g,opt,protocol_hash):
    p=directory/'checkpoint.pt'
    if not p.exists():return 0,[]
    assert sha(p)==read(directory/'checkpoint.json')['sha256']
    state=torch.load(p,map_location='cpu',weights_only=False)
    assert state['protocol_sha256']==protocol_hash
    g.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
    random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch']);torch.cuda.set_rng_state_all(state['rng_cuda'])
    print('FULL_STATIC_CHECKPOINT_RESTORED',state['iteration'],flush=True)
    return state['iteration'],state['camera_stack']


@torch.no_grad()
def evaluate(render,g,step,output,split='val',save_images=True):
    meta=read(ROOT/f'data/derived/kinematic_resplit/transforms_{split}.json')
    rows=[]
    for frame in meta['frames']:
        n=int(Path(frame['file_path']).stem)
        target,pos,neg=image_data(frame,1600)
        cam=camera(frame,meta,1600);p=render_static(render,cam,g)
        rgb=p['render'].clamp(0,1);alpha=p['alpha'].squeeze(0)
        gt=torch.tensor(target,device='cuda').permute(2,0,1);positive=torch.tensor(pos,device='cuda');negative=torch.tensor(neg,device='cuda')
        mse=(rgb[:,positive]-gt[:,positive]).square().mean()
        rows.append({'frame':n,'candidate_static_PSNR':float(-10*torch.log10(mse.clamp_min(1e-12))),
                     'candidate_static_alpha_mean':float(alpha[positive].mean()),
                     'candidate_static_alpha_below_05':float((alpha[positive]<.5).float().mean()),
                     'background_alpha_mean':float(alpha[negative].mean())})
        if save_images and n in [4,85,189,196,197]:
            output.mkdir(parents=True,exist_ok=True)
            im=(rgb.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
            Image.fromarray(im).save(output/f'{split}_{n:04d}_step{step:05d}.png')
    result={'iteration':step,'split':split,'count':len(rows),'frames':rows,
            'metrics':{k:float(np.mean([row[k] for row in rows])) for k in rows[0] if k!='frame'},
            'candidate_masks_are_conditional_not_GT_geometry':True}
    write(REPORT/f'S1_{split}_{step:05d}.json',result)
    print('STATIC_EVALUATION',step,split,result['metrics'],flush=True)
    return result


def preflight(render,g,opt,directory,hash_value):
    meta=read(DATA/'transforms_train.json');frame=next(f for f in meta['frames'] if f['file_path'].endswith('0086'))
    target,pos,neg=image_data(frame,1600);cam=camera(frame,meta,1600)
    package=render_static(render,cam,g)
    from output.short_clip_diagnostic.camera_math import w2c,intrinsic,project
    uv,_=project(g.get_xyz.detach().cpu().numpy(),w2c(frame['transform_matrix'],'metadata_cv'),
                 intrinsic(meta['camera_angle_x'],'metadata',1600,900))
    visible=package['visibility_filter'].detach().cpu().numpy()
    projection_error=float(np.max(np.abs(uv[visible]-package['proj_means_2D'].detach().cpu().numpy()[visible])))
    assert projection_error<.01,projection_error
    p=torch.tensor(pos,device='cuda');n=torch.tensor(neg,device='cuda');gt=torch.tensor(target,device='cuda').permute(2,0,1)
    loss,_=masked_objective(package['render'],package['alpha'],gt,p,n)
    out=ROOT/'output/staged_reconstruction/S1_static';out.mkdir(parents=True,exist_ok=True)
    Image.fromarray((package['render'].detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(out/'initial_t86.png')
    loss.backward()
    gradients={name:float(parameter.grad.abs().sum()) for name,parameter in g.named_parameters()}
    assert all(math.isfinite(x) and x>0 for x in gradients.values()),gradients
    before={n:p.detach().clone() for n,p in g.named_parameters()};opt.step();g.constrain()
    assert all(not torch.equal(before[n],p) for n,p in g.named_parameters())
    # Parameter bank must produce the exact same output in the original class.
    from scene_PIDG.gaussian_model import GaussianModel
    original=GaussianModel(3);original.active_sh_degree=3
    for name,parameter in g.named_parameters():setattr(original,name,parameter)
    with torch.no_grad():
        x=render_static(render,cam,g);y=render_static(render,cam,original)
        assert torch.equal(x['render'],y['render']) and torch.equal(x['alpha'],y['alpha'])
    # Recover an actual optimizer state; restored next update is finite.
    save(directory,g,opt,1,[0],hash_value)
    g2=StaticGaussian(RUNS/read(REPORT/'S1_protocol.json')['initialization_file']);o2=optimizer_for(g2)
    step,stack=restore(directory,g2,o2,hash_value);assert step==1 and stack==[0]
    assert all(torch.equal(v,g2.state_dict()[n]) for n,v in g.state_dict().items())
    opt.zero_grad(set_to_none=True)
    write(REPORT/'S1_preflight.json',{'passed':True,'native_resolution':[1600,900],
        'parameter_gradient_L1':gradients,'exact_original_renderer_adapter_parity':True,
        'optimizer_RNG_sampler_recovery':True,'no_dynamic_parameters':True,
        'protocol_sha256':hash_value,'native_projection_max_abs_px':projection_error,
        'baseline_training_source_modified':False})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-static-stage',action='store_true')
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--stop-after',type=int)
    parser.add_argument('--evaluate-only',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if not args.enable_static_stage:raise SystemExit('S1 is opt-in; pass --enable-static-stage')
    protocol=read(REPORT/'S1_protocol.json');assert protocol['static_only'] and not protocol['physics']
    torch.set_num_threads(4);random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
    render=setup();directory=RUNS/('_preflight' if args.preflight_only else '_smoke_v'+str(protocol['version']) if args.smoke else 'S1_static');directory.mkdir(parents=True,exist_ok=True)
    initialization=RUNS/protocol['initialization_file']
    assert sha(initialization)==read(REPORT/'data_receipt.json')['initialization_sha256']
    g=StaticGaussian(initialization);opt=optimizer_for(g);hash_value=sha(REPORT/'S1_protocol.json')
    if args.preflight_only:preflight(render,g,opt,directory,hash_value);return
    assert read(REPORT/'S1_preflight.json')['passed']
    assert read(REPORT/'S1_preflight.json')['protocol_sha256']==hash_value
    if not args.smoke:
        from .freeze import verify
        verify()
    start,stack=restore(directory,g,opt,hash_value)
    output=ROOT/'output/staged_reconstruction'/('_smoke' if args.smoke else 'S1_static');output.mkdir(parents=True,exist_ok=True)
    if args.evaluate_only:
        evaluate(render,g,start,output,'val');evaluate(render,g,start,output,'test');return
    meta=read(DATA/'transforms_train.json');cache={};cameras={};width=None;began=time.monotonic()
    write(RUNS/'state.json',{'stage':'S1','status':'training','pid':os.getpid(),'iteration':start})
    try:
        for step in range(start+1,protocol['stage_updates']+1):
            new_width=800 if step<=8000 else 1600
            if width!=new_width:cache.clear();cameras.clear();width=new_width
            if not stack:stack=random.sample(range(len(meta['frames'])),len(meta['frames']))
            index=stack.pop();frame=meta['frames'][index]
            if index not in cache:
                target,pos,neg=image_data(frame,width)
                cache[index]=(torch.tensor(target).permute(2,0,1),torch.tensor(pos),torch.tensor(neg))
                cameras[index]=camera(frame,meta,width)
            gt,p,n=[x.cuda() for x in cache[index]]
            opt.zero_grad(set_to_none=True);pkg=render_static(render,cameras[index],g)
            loss,terms=masked_objective(pkg['render'],pkg['alpha'],gt,p,n)
            loss=loss+.01*(g._xyz-g.initial_xyz).square().mean()
            assert bool(torch.isfinite(loss));loss.backward()
            assert all(p.grad is None or bool(p.grad.isfinite().all()) for p in g.parameters())
            opt.step();g.constrain()
            if step%20==0:
                event={'iteration':step,'frame':frame['file_path'],'width':width,'loss':float(loss),
                       **{k:float(v) for k,v in terms.items()}}
                with (directory/'losses.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            if step%200==0:
                write(RUNS/'state.json',{'stage':'S1','status':'training','pid':os.getpid(),'iteration':step})
                print('STATIC_TRAIN',step,'loss',float(loss),'elapsed',round(time.monotonic()-began,1),flush=True)
            if step%protocol['checkpoint_interval']==0:save(directory,g,opt,step,stack,hash_value)
            if step%protocol['validation_interval']==0:evaluate(render,g,step,output,save_images=False)
            if (RUNS/'stop.request').exists() or args.stop_after and step>=args.stop_after:
                save(directory,g,opt,step,stack,hash_value)
                write(RUNS/'state.json',{'stage':'S1','status':'saved_stopped','iteration':step});return
        save(directory,g,opt,protocol['stage_updates'],stack,hash_value)
        val=evaluate(render,g,protocol['stage_updates'],output,'val');evaluate(render,g,protocol['stage_updates'],output,'test')
        from .freeze import verify
        verify()
        write(RUNS/'state.json',{'stage':'S1','status':'evaluated_pending_review','iteration':protocol['stage_updates'],'validation':val['metrics']})
    except Exception as error:
        write(RUNS/'state.json',{'stage':'S1','status':'failed','error':str(error)});raise


if __name__=='__main__':
    import msvcrt
    RUNS.mkdir(parents=True,exist_ok=True)
    with (RUNS/'static_training.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
