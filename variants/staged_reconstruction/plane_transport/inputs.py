"""Read only the audited D6 train-derived plane metadata, never GT."""
import numpy as np
import torch
from ..prepare_data import ROOT

PARENT=ROOT/'runs/pidg_staged_reconstruction/S1_rgb_plane_warp/v1'


def load_parent(case,index):
    with np.load(PARENT/case/f'view_{index:03d}.npz') as a:
        return {key:a[key].copy() for key in ['uv','original','updated','coefficients','slope_admitted','eligible']}


def make_maps(case,height,width,device='cuda'):
    original=np.zeros((96,height,width),np.float32)
    slopes=np.full((96,height,width,2),np.nan,np.float32); admitted=np.zeros((96,height,width),bool)
    for i in range(96):
        p=load_parent(case,i); x,y=p['uv'].T
        original[i,y,x]=p['original'] # Never use updated source depth to replace an unchecked intercept.
        slopes[i,y,x]=p['coefficients'][:,:2]
        admitted[i,y,x]=p['slope_admitted']&p['eligible']
    return {key:torch.tensor(value,device=device) for key,value in [('original',original),('slopes',slopes),('admitted',admitted)]}
