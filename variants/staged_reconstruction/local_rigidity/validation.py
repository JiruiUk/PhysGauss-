"""Original numerical warmup gates, isolated reports and native images."""
import numpy as np
from PIL import Image
import torch
from ..prepare_data import ROOT, REPORT, DATA, read, write
from ..train_static import camera, PIPE
from ..model import observation_masks
from ..motion_integrity.evaluate import pixels
from .prepare import EXP


@torch.no_grad()
def validate_warmup(renderer,scene,base):
    from ..union import evaluate as original
    previous=original.UNION_REPORT
    original.UNION_REPORT=EXP
    try: result=original.evaluate(renderer,scene,'warmup',16000,'val',1600)
    finally: original.UNION_REPORT=previous
    # Frozen static-only reconstruction: no dynamic population may degrade it
    # beyond the same original 1dB guardrail on common static image support.
    meta=read(ROOT/'data/derived/kinematic_resplit/transforms_val.json')
    values=[]
    from ..stage2.train import image
    for frame in meta['frames']:
        rgba=image(frame,1600)
        gt=torch.tensor(rgba[...,:3]*rgba[...,3:4],device='cuda').permute(2,0,1)
        pred=renderer(camera(frame,meta,1600),scene.static,PIPE,scene.centers.new_zeros(3),0.,0.,0.)['render'].clamp(0,1)
        static,_=observation_masks(rgba,16)
        mask=torch.tensor(static,device='cuda')
        values.append(float(-10*torch.log10((pred[:,mask]-gt[:,mask]).square().mean().clamp_min(1e-12))))
    metrics=result['metrics']; gate=base['warmup_admission']
    checks={'dynamic_PSNR':metrics['dynamic_PSNR']>=gate['dynamic_PSNR_min'],
            'dynamic_alpha':metrics['dynamic_alpha']>=gate['dynamic_only_alpha_min'],
            'static_preservation':metrics['static_PSNR']>=float(np.mean(values))-gate['static_PSNR_drop_max']}
    train=read(DATA/'transforms_train.json'); by_id={int(f['file_path'].split('/')[-1]):f for f in train['frames']}
    out=ROOT/'output/staged_reconstruction/local_rigidity/warmup'; out.mkdir(parents=True,exist_ok=True)
    for cam_id in [86,188]:
        cam=camera(by_id[cam_id],train,1600)
        for number in [2,43,86,146,200]:
            Image.fromarray(pixels(scene.render(renderer,cam,(number-1)/199)['render'])).save(out/f'fixed{cam_id}_t{number:04d}_native.png')
    return {'passed':all(checks.values()),'numeric_gates':checks,'metrics':metrics,
            'static_only_PSNR':float(np.mean(values)),'native_images_reviewed':False,
            'admission_scope':'Conditional original6k physics comparison only; no geometry success claim'}
