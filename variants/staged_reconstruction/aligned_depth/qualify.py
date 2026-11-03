"""Robust inverse-depth calibration against frozen static support, train only."""
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import binary_erosion
import torch
from ..prepare_data import ROOT, REPORT, DATA, read, write
from ..train_static import camera, PIPE
from ..stage2.train import image
from ..union.core import dynamic_mask
from ..motion_integrity.prepare import load_parent


def fit(x, inverse_z):
    center, spread = np.median(x), np.quantile(x,.9)-np.quantile(x,.1)
    design = np.stack([(x-center)/max(spread,1e-6),np.ones_like(x)],-1)
    weights = np.ones(len(x))
    for _ in range(20):
        solution = np.linalg.lstsq(design*weights[:,None]**.5,inverse_z*weights**.5,rcond=None)[0]
        residual = design@solution-inverse_z
        scale = max(1.4826*np.median(np.abs(residual-np.median(residual))),1e-5)
        weights = np.minimum(1.,1.345*scale/np.maximum(np.abs(residual),1e-12))
    return float(solution[0]/max(spread,1e-6)),float(solution[1]-solution[0]*center/max(spread,1e-6))


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    renderer,scene,_,_,_=load_parent()
    meta=read(DATA/'transforms_train.json')
    frames={int(Path(f['file_path']).stem):f for f in meta['frames']}
    rows=[]
    for number in [2,43,86,146,188,200]:
        frame=frames[number]; rgba=image(frame,800); cam=camera(frame,meta,800)
        r,g,b=rgba[...,:3].transpose(2,0,1)
        mask=binary_erosion((rgba[...,3]>.95)&(b>r*1.3+.02)&(b>g*1.05),iterations=3)
        p=renderer(cam,scene.static,PIPE,scene.centers.new_zeros(3),0.,0.,0.)
        alpha=p['alpha'].squeeze().cpu().numpy(); z=p['depth'].squeeze().cpu().numpy()/np.maximum(alpha,1e-6)
        mask &= (alpha>.99)&(z>1)&(z<15)
        pseudo=cv2.resize(np.load(ROOT/f'data/PIDG/rubber_duck/depth_distill/{number:04d}.npy'),(800,450),interpolation=cv2.INTER_AREA)
        y,x=np.nonzero(mask); chosen=np.arange(len(x))[::3]
        a,bias=fit(pseudo[y[chosen],x[chosen]],1/z[y[chosen],x[chosen]])
        predicted=1/np.maximum(a*pseudo+bias,1e-6)
        held=np.arange(len(x))%3!=0
        residual=np.abs(predicted[y[held],x[held]]-z[y[held],x[held]])
        dm=binary_erosion(dynamic_mask(rgba),iterations=2)
        rows.append({'frame':number,'a':a,'b':bias,'static_support':len(x),
                     'within_training_frame_held_pixel_depth_error_q50_q90':np.quantile(residual,[.5,.9]).tolist(),
                     'dynamic_target_depth_quantiles':np.quantile(predicted[dm],[.01,.1,.5,.9,.99]).tolist(),
                     'dynamic_fraction_outside_static_pseudo_range':float(((pseudo[dm]<np.quantile(pseudo[mask],.01))|(pseudo[dm]>np.quantile(pseudo[mask],.99))).mean())})
        print('ALIGNED_DEPTH_QUALIFICATION',rows[-1],flush=True)
    write(REPORT/'aligned_depth/qualification.json',{'rows':rows,'training_only':True,'depth_is_GT':False,
          'optimizer_updates':0,'calibration_is_model_conditioned':True})


if __name__=='__main__': main()
