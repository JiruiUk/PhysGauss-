"""Read-only J8 native evidence; training reprojection is never called GT."""
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from variants.staged_reconstruction.j7_long_memory import runtime as a
from variants.staged_reconstruction.i10_track_train.core import observations
from variants.staged_reconstruction.i10_tracking.io import ROOT,DATA,read,write,sha
from variants.staged_reconstruction.train_static import camera,render_static,PIPE
from experiments.staged_reconstruction.RubberDuck_joint_pilot import evaluate as helper

r=a.r
LABELS=['J7_parent','rgb_control','temporal_memory']


def load_models():
    from experiments.staged_reconstruction.RubberDuck_J7_long_memory.audit import audit
    p=read(a.BASE/'protocol.json');audits=audit()
    renderer,parent,_,_,_=r.load('rgb_control');models={'J7_parent':parent}
    for arm in p['arms']:
        renderer,scene,opts,_,_=r.load(arm)
        r.restore(a.RUNS/'formal'/arm/'checkpoint.pt',scene,opts);models[arm]=scene
    write(a.REPORT/'independent_audit.json',audits)
    return renderer,models,audits


@torch.no_grad()
def validation(renderer,models):
    from utils.loss_utils import ssim
    meta=read(ROOT/'data/derived/kinematic_resplit/transforms_val.json');assert len(meta['frames'])==30
    rows=[]
    for index,frame in enumerate(meta['frames']):
        assert not r.stopped();cam=camera(frame,meta,1600);target=r.frame_target(frame)
        row=dict(frame=frame['file_path'])
        for name,scene in models.items():
            pkg=scene.render(renderer,cam,frame['time']);pred=pkg['render'].clamp(0,1)
            row[name]=dict(PSNR=float(-10*torch.log10((pred-target).square().mean().clamp_min(1e-12))),
                SSIM=float(ssim(pred,target)),L1=float((pred-target).abs().mean()))
        rows.append(row);print('J8_VALIDATION',index+1,30,flush=True)
    means={n:{k:float(np.mean([v[n][k] for v in rows])) for k in ['PSNR','SSIM','L1']} for n in LABELS}
    return dict(frames=rows,means=means,all30_native1600=True,semantic_masks_used=False,test_not_read=True,
        PSNR_nonregression_gate=means['temporal_memory']['PSNR']>=means['rgb_control']['PSNR']-.3,
        parent_nonregression_gate=means['temporal_memory']['PSNR']>=means['J7_parent']['PSNR']-.3,
        independent_geometry_GT=False)


from experiments.staged_reconstruction.RubberDuck_J7_long_memory.diagnostics import diagnostics

@torch.no_grad()
def track_diagnostic(models):
    from variants.staged_reconstruction.i10_tracking.core import project_world
    result={}
    for name,scene in models.items():
        total=missing=0;fit=[];held=[]
        for memory in a.memory.MEMORIES:
            frames=memory['frames'];row=memory['binding']['row']
            xyz=np.array([scene.dynamic_at(a.memory.META['frames'][f]['time']).get_xyz[row].cpu().numpy() for f in frames])
            uv,z=project_world(xyz[:,None],a.memory.ASSET['views'][frames],a.memory.ASSET['k'])
            error=np.linalg.norm(uv[:,0]-memory['uv'],axis=-1);good=np.isfinite(error)&(z[:,0]>.01)
            total+=len(frames);missing+=int((~good).sum())
            fit += [float(e) for f,e,g in zip(frames,error,good) if not f%2 and g]
            held += [float(e) for f,e,g in zip(frames,error,good) if f%2 and g]
        result[name]=dict(total=total,missing=missing,fit_mean_px=float(np.mean(fit)),held_mean_px=float(np.mean(held)))
    return dict(all_original_bound_memories_unfiltered=True,held_RGB_trained=True,independent_3D_GT=False,models=result)


def freeze():
    files=[Path(__file__),a.BASE/'audit.py',a.BASE/'diagnostics.py',
        ROOT/'experiments/staged_reconstruction/RubberDuck_joint_pilot/evaluate.py',
        ROOT/'data/derived/kinematic_resplit/transforms_val.json']
    files += [DATA/(f['file_path']+'.png') for f in read(files[-1])['frames']]
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    path=a.REPORT/'evaluation_freeze.json'
    if path.exists():assert read(path)==hashes
    else:write(path,hashes)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-surface-evaluation',action='store_true')
    parser.add_argument('--phase',choices=['evaluate','video'],default='evaluate');args=parser.parse_args()
    if not args.enable_surface_evaluation:raise SystemExit('Evaluation opt-in')
    a.install(True);r.verify();assert not r.stopped();torch.set_num_threads(4);cv2.setNumThreads(2);a.OUT.mkdir(parents=True,exist_ok=True);freeze()
    with r.lock('evaluation.lock'):
        renderer,models,audits=load_models()
        if args.phase=='evaluate':
            assert not (a.REPORT/'evaluation.json').exists()
            result=dict(checkpoint_audits=audits,validation=validation(renderer,models),diagnostics=diagnostics(renderer,models),
                training_track_diagnostic=track_diagnostic(models),plys=helper.plys(models),source_sha256=sha(Path(__file__)))
            r.verify();write(a.REPORT/'evaluation.json',result);print('J8_EVALUATION_COMPLETE',result['validation']['means'],flush=True)
        else:
            assert not (a.REPORT/'videos.json').exists() and (a.REPORT/'evaluation.json').exists()
            result=helper.videos(renderer,models);r.verify();write(a.REPORT/'videos.json',dict(videos=result,source_sha256=sha(Path(__file__))))

if __name__=='__main__':main()
