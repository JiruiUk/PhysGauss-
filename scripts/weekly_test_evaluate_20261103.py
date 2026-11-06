"""Frozen-model, common-GT test evaluation for the weekly report. No optimization."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tmp/pdfs/weekly_20261103/test_compare'
MODELS=['H3','S0','S1','S2','S3','S4','S5','S6']
CHECKPOINTS={
    'H3':'runs/pidg_integrity_tuning/H3_combined/rubber_duck/checkpoints/resume.pt',
    'S0':'runs/pidg_scratch_integrity/pilot/S0_scratch_control/rubber_duck/checkpoints/resume.pt',
    'S1':'runs/pidg_staged_reconstruction/S1_static/checkpoint.pt',
    'S2':'runs/pidg_staged_reconstruction/S2_geometry_recovery/S2_objects/checkpoint.pt',
    **{s:f'runs/pidg_staged_reconstruction/{s}/formal/checkpoint.pt' for s in ['S3','S4','S5','S6']}}
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,j):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')

def worker(model):
    import numpy as np
    from PIL import Image
    import torch
    sys.path.insert(0,str(ROOT))
    torch.set_num_threads(4)
    out=OUT/'evaluation'/f'{model}.json'
    checkpoint=ROOT/CHECKPOINTS[model]; before=sha(checkpoint)
    metadata=ROOT/'data/derived/kinematic_resplit/transforms_test.json'
    frames=read(metadata)['frames']; ids=[int(Path(f['file_path']).stem) for f in frames]
    assert ids==list(range(1,201,4))
    for n,f in zip(ids,frames):assert abs(f['time']-(n-1)/199)<1e-10
    protocol={'resolution':[1600,900],
        'GT':'PIL RGBA bilinear resize to1600x900, divide255, black-composite in float32',
        'prediction':'clip[0,1], round8bit, divide255; no image repair or postprocessing',
        'PSNR':'mean of per-frame -10log10 full-frame RGB MSE; floor1e-12',
        'SSIM':'PIDG utils.loss_utils.ssim; unchanged11px Gaussian window',
        'LPIPS':'local standard VGG v0.1 calibrated weights, inputs[-1,1]',
        'static_mask':'original S1 observation_masks(rgba,16)',
        'dynamic_mask':'original staged evaluator: alpha>.95,r>.25,g>.18,b<.8*min(r,g)',
        'S1_scope':'static-only; full-frame/dynamic metrics intentionally null',
        'baseline':'H3_combined final50k, prior validation-promoted reference; never per-frame oracle',
        'test_frame_ids':ids,'metadata_sha256':sha(metadata),'script_sha256':sha(Path(__file__)),
        'selection':'no new checkpoint/model selection or tuning using these results'}
    if out.exists():
        old=read(out)
        assert old['checkpoint_sha256']==before and old['protocol']==protocol
        assert all(sha(ROOT/rel)==value for rel,value in old['source_sha256'].items())
        print('ALREADY_VERIFIED',model,flush=True);return
    os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
    if model=='S0':
        directory=checkpoint.parents[1]
        sys.path.insert(0,str(ROOT/'variants/association_training/code'))
        os.environ['PIDG_METRIC_CONFIG']=str(directory/'metric_config.json')
        from variants.color_repair.color import _read
        _read.cache_clear()
        from variants.association_training.evaluate import load,render_call
        from variants.shape_coherence.observations import camera
        from variants.flow_comparison.temporal import warp_at
        g,deform,proof=load(directory);assert proof['iteration']==50000
        def predict(frame,meta):
            return render_call(camera(frame,meta,1600),g,warp_at(deform,g.get_xyz.detach(),frame['time']))['render']
    elif model=='H3':
        sys.path.insert(0,str(ROOT/'variants/color_repair/code'))
    else:
        os.environ.pop('PIDG_METRIC_CONFIG',None)
        from variants.staged_reconstruction.train_static import setup,camera,render_static
        renderer=setup()
        if model=='S1':
            from variants.staged_reconstruction.static_gaussian import StaticGaussian
            scene=StaticGaussian(ROOT/'runs/pidg_staged_reconstruction/static_metadata_initialization.npz')
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert state['iteration']==12000
            scene.load_state_dict(state['model']);scene.requires_grad_(False)
            def predict(frame,meta):return render_static(renderer,camera(frame,meta,1600),scene)['render']
        else:
            if model=='S2':
                from variants.staged_reconstruction.geometry_recovery.model import BoundedScene
                scene=BoundedScene(read(ROOT/'experiments/staged_reconstruction/S2_geometry_recovery/S2_protocol.json'))
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                assert state['iteration']==16000
                scene.load_state_dict(state['model'])
            else:
                from variants.staged_reconstruction.stage4.bind import load_physics_state
                scene,state,_=load_physics_state(checkpoint)
                assert state['iteration']==(6000 if model=='S3' else 4000)
            scene.requires_grad_(False)
            def predict(frame,meta):return scene.render(renderer,camera(frame,meta,1600),frame['time'])['render']
    from utils.loss_utils import ssim
    from variants.staged_reconstruction.model import observation_masks
    import lpips
    vgg=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
    linear=Path(lpips.__file__).parent/'weights/v0.1/vgg.pth'
    assert vgg.exists() and linear.exists(),'Use existing local weights only'
    metric=lpips.LPIPS(net='vgg',version='0.1').cuda().eval().requires_grad_(False)
    source_hashes={str(metadata.relative_to(ROOT)):sha(metadata),
        str(vgg):sha(vgg),str(linear):sha(linear)}
    rows=[];meta=read(metadata)
    with torch.no_grad():
        for index,frame in enumerate(frames):
            n=ids[index]
            original=ROOT/f'data/PIDG/rubber_duck/train/{n:04d}.png'
            source_hashes[str(original.relative_to(ROOT))]=sha(original)
            with Image.open(original) as im:
                rgba=np.asarray(im.convert('RGBA').resize((1600,900),Image.Resampling.BILINEAR),dtype=np.float32)/255
            gt=torch.as_tensor(rgba[...,:3]*rgba[...,3:4],device='cuda').permute(2,0,1)[None]
            if model=='H3':
                source=ROOT/f'runs/pidg_integrity_tuning/H3_combined/rubber_duck/test/ours_50000/renders/{index:05d}.png'
                source_hashes[str(source.relative_to(ROOT))]=sha(source)
                with Image.open(source) as im:array=np.asarray(im.convert('RGB'),dtype=np.float32)/255
                assert array.shape==(900,1600,3)
                pred=torch.as_tensor(array,device='cuda').permute(2,0,1)[None]
            else:pred=(predict(frame,meta).clamp(0,1)*255).round()[None]/255
            assert pred.shape==gt.shape and pred.isfinite().all()
            png=OUT/'renders'/model/f'{n:04d}.png';png.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray((pred[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(png)
            source_hashes[str(png.relative_to(ROOT))]=sha(png)
            static,_=observation_masks(rgba,16)
            r,gc,b=rgba[...,:3].transpose(2,0,1)
            dynamic=(rgba[...,3]>.95)&(r>.25)&(gc>.18)&(b<.8*np.minimum(r,gc))
            def psnr(mask=None):
                delta=(pred-gt) if mask is None else (pred[0,:,torch.as_tensor(mask,device='cuda')]-gt[0,:,torch.as_tensor(mask,device='cuda')])
                return float(-10*torch.log10(delta.square().mean().clamp_min(1e-12)))
            row={'frame':n,'time':frame['time'],'PSNR':psnr() if model!='S1' else None,
                'SSIM':float(ssim(pred,gt)) if model!='S1' else None,
                'LPIPS':float(metric(pred*2-1,gt*2-1).mean()) if model!='S1' else None,
                'static_PSNR':psnr(static),'dynamic_PSNR':psnr(dynamic) if model!='S1' else None,
                'static_pixels':int(static.sum()),'dynamic_pixels':int(dynamic.sum()),
                'render':str(png.relative_to(ROOT))}
            rows.append(row)
            if (index+1)%10==0:print('TEST',model,index+1,'/50',flush=True)
    assert sha(checkpoint)==before
    fields=['PSNR','SSIM','LPIPS','static_PSNR','dynamic_PSNR']
    means={k:float(np.mean([row[k] for row in rows])) if rows[0][k] is not None else None for k in fields}
    write(out,{'model':model,'checkpoint':CHECKPOINTS[model],'checkpoint_sha256':before,
        'protocol':protocol,'captured_utc':datetime.now(timezone.utc).isoformat(),'frames':rows,'metrics':means,
        'checkpoint_unchanged_after':True,'training_or_optimizer_updates':0,'source_sha256':source_hashes,
        'weight_sha256':{'VGG':sha(vgg),'linear':sha(linear)},
        'render_source':'existing final50k PNG' if model=='H3' else 'read-only checkpoint inference'})
    print('COMPLETED',model,means,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS);a=p.parse_args()
    if a.model:worker(a.model);return
    OUT.mkdir(parents=True,exist_ok=True)
    for model in MODELS:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--model',model],cwd=ROOT,check=True)
    print('ALL_EIGHT_TEST_MODELS_COMPLETED',flush=True)
if __name__=='__main__':main()
