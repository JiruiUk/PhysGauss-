import torch
from variants.staged_reconstruction.stage4.tracks import advect


def test_sparse_track_translation_has_explicit_FB_consistency():
    f=torch.zeros(24,32,2);f[...,0]=2;b=-f
    labels=torch.zeros(24,32,dtype=torch.long);p=torch.tensor([[6.,7.],[10.,12.]])
    q,valid=advect(p,f,b,labels,labels,0,0)
    assert valid.all() and torch.equal(q,p+torch.tensor([2.,0.]))
    _,bad=advect(p,f,f,labels,labels,0,0);assert not bad.any()


def test_unknown_or_cross_component_track_never_acquires_an_identity():
    f=torch.zeros(24,32,2);labels=torch.zeros(24,32,dtype=torch.long);p=torch.tensor([[6.,7.]])
    _,a=advect(p,f,f,labels,labels,-1,0);_,b=advect(p,f,f,labels,labels,0,1)
    assert not a.any() and not b.any()
