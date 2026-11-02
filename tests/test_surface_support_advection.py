import numpy as np
import torch
from output.surface_support.core import propagate


def test_raw_flow_direction_and_pixel_units():
    labels=np.full((12,12),-1);labels[4:8,3:6]=2
    f=torch.zeros(12,12,2);f[...,0]=2;b=-f
    target=np.full((12,12),-1);target[4:8,5:8]=0
    moved,amb=propagate(labels,target,f,b)
    assert (moved[4:8,5:8]==2).all() and not amb
    assert (moved[:,:5]==-1).all()


def test_out_of_bounds_is_unknown():
    labels=np.zeros((12,12),dtype=int);f=torch.zeros(12,12,2);f[...,0]=30
    moved,_=propagate(labels,np.zeros((12,12),dtype=int),f,-f)
    assert (moved==-1).all()
