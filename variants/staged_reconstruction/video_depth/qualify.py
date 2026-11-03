"""Separate-process static geometry qualification; learned depth is never GT."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
import torch
from ..prepare_data import ROOT,REPORT,DATA,read,write,sha
from ..train_static import camera,PIPE
from ..stage2.train import image
from ..union.core import dynamic_mask
from ..motion_integrity.prepare import load_parent
from ..aligned_depth.qualify import fit
from .fetch import EXP
from .core import ANCHORS,GATES,check


@torch.no_grad()
def main():
    assert not (EXP/'qualification.json').exists(),'Preserve qualification versions'
    torch.set_num_threads(4)
    protocol=read(EXP/'pilot_protocol.json'); assert protocol['quality_gates']==GATES
    inference=read(EXP/'inference.json')
    renderer,scene,_,_,_=load_parent()
    meta=read(DATA/'transforms_train.json'); frames={int(Path(f['file_path']).stem):f for f in meta['frames']}
    rows=[]; cache={}; ratios=[]
    for number in ANCHORS:
        receipt=inference['receipts'][str(number)]; path=ROOT/receipt['path']; assert sha(path)==receipt['sha256']
        with np.load(path) as data:
            where=list(data['frame_ids']).index(number); predicted=data['depth'][where].copy()
        rgba=image(frames[number],800); r,g,b=rgba[...,:3].transpose(2,0,1)
        mask=binary_erosion((rgba[...,3]>.95)&(b>r*1.3+.02)&(b>g*1.05),iterations=3)
        package=renderer(camera(frames[number],meta,800),scene.static,PIPE,scene.centers.new_zeros(3),0.,0.,0.)
        alpha=package['alpha'].squeeze().cpu().numpy()
        z=package['depth'].squeeze().cpu().numpy()/np.maximum(alpha,1e-6)
        mask &= (alpha>.99)&(z>1)&(z<15)&np.isfinite(predicted)&(predicted>1e-5)
        y,x=np.nonzero(mask); fit_ids=np.arange(len(x))[::3]; held_ids=np.flatnonzero(np.arange(len(x))%3!=0)
        assert len(fit_ids)>100 and len(held_ids)>100
        ratios.append(z[y[fit_ids],x[fit_ids]]/predicted[y[fit_ids],x[fit_ids]])
        dm=binary_erosion(dynamic_mask(rgba),iterations=2)
        pseudo=cv2.resize(np.load(ROOT/f'data/PIDG/rubber_duck/depth_distill/{number:04d}.npy'),(800,450),interpolation=cv2.INTER_AREA)
        a,bias=fit(pseudo[y[fit_ids],x[fit_ids]],1/z[y[fit_ids],x[fit_ids]])
        old=1/np.maximum(a*pseudo+bias,1e-6)
        cache[number]=(predicted,z,mask,dm,rgba,y,x,held_ids,old,pseudo)
    scale=float(np.median(np.concatenate(ratios))); assert np.isfinite(scale) and scale>0
    out=ROOT/'output/staged_reconstruction/video_depth/pilot'; out.mkdir(parents=True,exist_ok=True)
    for number in ANCHORS:
        predicted,z,mask,dm,rgba,y,x,held,old,pseudo=cache[number]
        depth=scale*predicted
        residual=np.abs(depth[y[held],x[held]]-z[y[held],x[held]])
        q=np.quantile(residual,[.5,.9])
        lower,upper=np.quantile(predicted[mask],[.01,.99])
        extrap=float(((predicted[dm]<lower)|(predicted[dm]>upper)).mean())
        old_error=np.abs(old[y[held],x[held]]-z[y[held],x[held]])
        row=dict(frame=number,median=float(q[0]),p90=float(q[1]),static_support=len(x),
            dynamic_extrapolation_fraction=extrap,per_frame_scale_diagnostic=float(np.median(z[mask]/predicted[mask])),
            old_per_frame_affine_inverse_depth_median_p90=np.quantile(old_error,[.5,.9]).tolist(),
            dynamic_depth_quantiles=np.quantile(depth[dm],[.01,.5,.99]).tolist())
        row['passed']=check(row); rows.append(row)
        panels=[(rgba[...,:3]*255).clip(0,255).astype(np.uint8)]
        for values in [depth,old]:
            colored=cv2.applyColorMap(((values-4)/4*255).clip(0,255).astype(np.uint8),cv2.COLORMAP_TURBO)[...,::-1]
            colored[rgba[...,3]<.5]=0; panels.append(colored)
        error=np.zeros_like(rgba[...,:3],dtype=np.uint8)
        error[mask,0]=(np.minimum(np.abs(depth[mask]-z[mask])/.3,1)*255).astype(np.uint8)
        panels.append(error)
        sheet=Image.new('RGB',(1600,952),'#151821')
        for index,(panel,label) in enumerate(zip(panels,['Train RGB','VDA scaled depth (4..8 scene units)','Old per-frame aligned pseudo-depth (4..8)','Static abs error: red at >=.3 scene units'])):
            offset=((index%2)*800,(index//2)*476)
            sheet.paste(Image.fromarray(panel),(offset[0],offset[1]+26)); ImageDraw.Draw(sheet).text((offset[0]+6,offset[1]+5),label,fill='white')
        sheet.save(out/f'anchor{number:04d}.png')
        print('VIDEO_DEPTH_QUALIFICATION',row,flush=True)
    write(EXP/'qualification.json',{'rows':rows,'global_metric_to_scene_scale':scale,'static_qualification_passed':all(r['passed'] for r in rows),
        'quality_gates':GATES,'static_reference_is_frozen_model_conditioned_not_GT':True,
        'irregular_training_time_spacing':True,'reduced_input_size':280,
        'training_admitted':False,'dynamic_geometry_and_correspondence_not_independently_validated':True,
        'optimizer_updates':0,'test_or_val_RGB_used':False,'PIDG_training_changed':False})


if __name__=='__main__': main()
