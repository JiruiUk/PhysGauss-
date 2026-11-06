"""Independent read-only four-model native evaluation; no mask-based selection."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from variants.staged_reconstruction.colorfree_joint import runtime as a
from variants.staged_reconstruction.prepare_data import ROOT,DATA,read,write,sha
from variants.staged_reconstruction.train_static import camera
from experiments.staged_reconstruction.RubberDuck_joint_pilot import evaluate as old
from experiments.staged_reconstruction.RubberDuck_colorfree_joint.audit import audit
from experiments.staged_reconstruction.RubberDuck_colorfree_joint.diagnostics import diagnostics

LABELS=['J6_parent','direct_fusion','old_static_control','colorfree_static']


def models():
    r=a.install(True);scenes={}
    renderer,scene,_,_,_=r.load('old_static_control');scenes['J6_parent']=scene
    renderer,scene,_,_,_=r.load('colorfree_static');scenes['direct_fusion']=scene
    for arm in read(a.BASE/'protocol.json')['arms']:
        renderer,scene,opts,_,_=r.load(arm)
        r.restore(a.RUNS/'formal'/arm/'checkpoint.pt',scene,opts);scenes[arm]=scene
    return renderer,scenes


@torch.no_grad()
def validation(renderer,scenes):
    from utils.loss_utils import ssim
    r=a.r;meta=read(ROOT/'data/derived/kinematic_resplit/transforms_val.json');assert len(meta['frames'])==30
    rows=[]
    for index,frame in enumerate(meta['frames']):
        assert not r.stopped();target=r.frame_target(frame);cam=camera(frame,meta,1600);row=dict(frame=frame['file_path'])
        for name,scene in scenes.items():
            pkg=scene.render(renderer,cam,frame['time']);pred=pkg['render'].clamp(0,1)
            row[name]=dict(PSNR=float(-10*torch.log10((pred-target).square().mean().clamp_min(1e-12))),
                           SSIM=float(ssim(pred,target)),L1=float((pred-target).abs().mean()))
        rows.append(row);print('J7_VALIDATION',index+1,30,flush=True)
    means={n:{k:float(np.mean([q[n][k] for q in rows])) for k in rows[0][n]} for n in LABELS}
    delta=means['colorfree_static']['PSNR']-means['old_static_control']['PSNR']
    parent_delta=means['colorfree_static']['PSNR']-means['J6_parent']['PSNR']
    return dict(frames=rows,means=means,native_width=1600,all30_full_RGB=True,test_RGB_not_read=True,
                semantic_masks_used=False,independent_geometry_GT=False,candidate_minus_control_PSNR_dB=delta,
                candidate_minus_J6_PSNR_dB=parent_delta,PSNR_nonregression_gate=delta>=-.3 and parent_delta>=-.3,
                manual_geometry_cleanup_required=True)


def freeze():
    paths=[Path(__file__),a.BASE/'audit.py',a.BASE/'diagnostics.py',ROOT/'experiments/staged_reconstruction/RubberDuck_joint_pilot/evaluate.py',
           ROOT/'data/derived/kinematic_resplit/transforms_val.json']
    meta=read(paths[-1]);paths += [DATA/(f['file_path']+'.png') for f in meta['frames']]
    files={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    receipt=a.REPORT/'evaluation_freeze.json'
    if receipt.exists():assert read(receipt)==files
    else:write(receipt,files)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-colorfree-evaluation',action='store_true')
    p.add_argument('--phase',choices=['evaluate','video'],default='evaluate');args=p.parse_args()
    if not args.enable_colorfree_evaluation:raise SystemExit('Evaluation is opt-in')
    r=a.install(True);assert not r.stopped();r.verify();torch.set_num_threads(4);old.LABELS=LABELS
    a.OUT.mkdir(parents=True,exist_ok=True);freeze()
    with r.lock('evaluation.lock'):
        evidence=audit();write(a.REPORT/'independent_audit.json',evidence)
        renderer,scenes=models()
        if args.phase=='evaluate':
            assert not (a.REPORT/'evaluation.json').exists()
            result=dict(stage='J7',labels={'J6_parent':'retained J6','direct_fusion':'replacement before training',
                'old_static_control':'old static + matched2000','colorfree_static':'color-free static + matched2000'},
                checkpoint_audits=evidence,validation=validation(renderer,scenes),diagnostics=diagnostics(renderer,scenes),plys=old.plys(scenes))
            r.verify();write(a.REPORT/'evaluation.json',result);print('J7_NATIVE_EVALUATION_DONE',result['validation']['means'],flush=True)
        else:
            assert (a.REPORT/'evaluation.json').exists() and not (a.REPORT/'videos.json').exists()
            videos=old.videos(renderer,scenes);r.verify();write(a.REPORT/'videos.json',dict(videos=videos))
            print('J7_TWO200_FRAME_H264_FULLY_DECODED',flush=True)


if __name__=='__main__':main()
