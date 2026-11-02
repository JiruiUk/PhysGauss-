import numpy as np
import torch
from output.surface_support.core import propagate,RenderProxy,full_composite,ratio,components


def test_metadata_twenty_is_not_full_compositing():
    a=np.full(25,.02);c=np.zeros(25);c[22]=1
    rgb,w=full_composite(a,c)
    assert rgb>0 and (w[:20]*c[:20]).sum()==0
    assert w.sum()<1 and w.sum()>w[:20].sum()


def test_proxy_never_mutates_model():
    class Model:get_opacity=torch.tensor([.5]);get_xyz=torch.tensor([[1.,2.,3.]])
    m=Model();p=RenderProxy(m,get_opacity=torch.zeros(1))
    assert p.get_opacity.item()==0 and m.get_opacity.item()==.5
    assert p.get_xyz is m.get_xyz


def test_unknown_never_assigned_to_neighbor():
    a=np.full((8,8),-1);a[2:5,2:4]=0
    f=torch.zeros(8,8,2);target=np.zeros((8,8),dtype=int)
    moved,amb=propagate(a,target,f,f)
    assert np.array_equal(a,moved) and not amb


def test_touching_component_ambiguity_is_rejected():
    a=np.full((8,8),-1);a[2:5,1:3]=0;a[2:5,5:7]=1
    moved,amb=propagate(a,np.zeros((8,8),dtype=int),torch.zeros(8,8,2),torch.zeros(8,8,2))
    assert (moved==-1).all() and amb==[0]


def test_lost_surface_cannot_reappear_as_known():
    a=np.full((8,8),-1);a[3,3]=0
    f=torch.zeros(8,8,2);gone,_=propagate(a,np.full((8,8),-1),f,f)
    returned,_=propagate(gone,np.zeros((8,8),dtype=int),f,f)
    assert (returned==-1).all()


def test_empty_support_ratio_is_not_success():
    assert ratio(0,0) is None and ratio(0,1)==0


def test_components_does_not_force_four_objects():
    a=np.zeros((20,20,4),dtype=np.uint8);a[4:15,4:15]=[255,230,20,255]
    c=components(a)
    assert set(np.unique(c))=={-1,0}
