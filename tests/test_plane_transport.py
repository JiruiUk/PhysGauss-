import subprocess
import sys
import numpy as np
import torch
from variants.staged_reconstruction.plane_transport.core import (world_point,source_plane,ray_plane_hit,transport,incumbent_plane,propose)
from variants.staged_reconstruction.adaptive_static.geometry import patch_features


def fixture(slope=0.,shift=.2,rotate=False):
    k=torch.tensor([[35.,0,31.5],[0,35.,31.5],[0,0,1.]])
    yy,xx=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
    pixels=torch.stack([xx,yy,torch.ones_like(xx)],-1).float()
    local=pixels@torch.linalg.inv(k).T
    views=torch.eye(4).repeat(7,1,1); views[:,0,3]=torch.tensor([0,shift,2*shift,3*shift,-shift,-2*shift,-3*shift])
    if rotate:
        a=torch.tensor(.13); r=torch.tensor([[a.cos(),0,a.sin()],[0,1,0],[-a.sin(),0,a.cos()]])
        views[1:,:3,:3]=r
    normal=torch.tensor([slope,0.,1.]); images=[]; depths=[]; slopes=[]
    for v in views:
        inv=torch.linalg.inv(v); origin=inv[:3,3]; rays=local@inv[:3,:3].T
        z=(4-normal@origin)/(rays*normal).sum(-1)
        xyz=origin+rays*z[...,None]; wx,wy=xyz[...,0],xyz[...,1]
        g=.45+.15*torch.sin(wx*7)+.1*torch.cos(wy*9+wx*3)
        images.append(torch.stack([g,.9*g,.8*g])); depths.append(z)
        nc=normal@v[:3,:3].T; dc=4+nc@v[:3,3]
        cov=(nc@torch.linalg.inv(k))/dc
        slopes.append(cov[:2].expand(64,64,2))
    images=torch.stack(images)
    maps={'original':torch.stack(depths),'slopes':torch.stack(slopes),'admitted':torch.ones(7,64,64,dtype=torch.bool)}
    return images,views,k,maps


def test_flags_off_returns_parent_object_without_sources():
    old=torch.tensor([4.8]); out,d=propose(None,old)
    assert out is old and not d['accepted'].any()


def test_cli_default_off():
    r=subprocess.run([sys.executable,'-m','variants.staged_reconstruction.plane_transport.infer'],capture_output=True,text=True)
    assert r.returncode!=0 and 'opt-in' in r.stderr


def test_source_plane_anchor_and_slanted_normal():
    _,views,k,maps=fixture(.4,rotate=True); uv=torch.tensor([[32.,30.],[29.,35.]])
    for j in [0,1,4]:
        z=maps['original'][j,uv[:,1].long(),uv[:,0].long()]
        s=maps['slopes'][j,uv[:,1].long(),uv[:,0].long()]
        anchor,normal=source_plane(uv,z,s,views[j],k)
        assert torch.allclose(.4*anchor[:,0]+anchor[:,2],torch.full_like(z,4),atol=1e-6)
        assert torch.allclose(normal/normal[:,2:],torch.tensor([[.4,0,1.]]).repeat(2,1),atol=1e-6)


def test_parallel_zero_and_behind_plane_reject():
    rays=torch.tensor([[0.,0,1.]]).repeat(3,1); origin=torch.zeros(3)
    anchor=torch.tensor([[0.,0,4.],[0.,0,4.],[0.,0,-4.]])
    normal=torch.tensor([[1.,0,0.],[0.,0,0.],[0.,0,1.]])
    _,valid,_=ray_plane_hit(origin,rays,anchor,normal)
    assert not valid.any()


def test_transport_rotation_recovers_reference_ray_depth_and_slope():
    _,views,k,maps=fixture(.4,rotate=True); uv=torch.tensor([[32.,30.]])
    # A source center close to the projected ray; use its exact analytic depth.
    p=world_point(uv,maps['original'][0,30,32,None],views[0],k)
    local=p@views[1,:3,:3].T+views[1,:3,3]; projected=local@k.T
    q=(projected[:,:2]/projected[:,2:]).round(); x,y=q.long()[0]
    z,s,valid,_=transport(uv,q,maps['original'][1,y,x,None],maps['slopes'][1,y,x,None],views[0],views[1],k,64,64,2,6)
    assert valid.all()
    assert torch.allclose(z,torch.tensor([4/(1+.4*.5/35)]),atol=1e-6)
    assert torch.allclose(s,torch.tensor([[.4/(4*35),0.]]),atol=1e-7)


def test_footprint_rejects_geometrically_correct_remote_plane():
    _,views,k,maps=fixture(); uv=torch.tensor([[32.,30.]])
    z,s,valid,projected=transport(uv,torch.tensor([[8.,30.]]),torch.tensor([4.]),torch.zeros(1,2),views[0],views[1],k,64,64,2,6)
    assert abs(z.item()-4)<1e-5 and not valid.any() and projected[0,0]>30


def test_transport_has_finite_nonzero_depth_gradients():
    _,views,k,maps=fixture(.4); uv=torch.tensor([[32.,30.]]); q=torch.tensor([[34.,30.]])
    depth=maps['original'][1,30,34,None].clone().requires_grad_(True)
    slopes=maps['slopes'][1,30,34,None].clone().requires_grad_(True)
    z,s,valid,_=transport(uv,q,depth,slopes,views[0],views[1],k,64,64,2,6)
    assert valid.all(); grads=torch.autograd.grad(z.sum()+s.sum(),[depth,slopes])
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)


def test_incumbent_fallback_is_comparison_only():
    _,views,k,_=fixture(); uv=torch.tensor([[32.,30.],[29.,35.]])
    slopes=torch.tensor([[float('nan'),0.],[1.,0.]])
    s,f,_,valid=incumbent_plane(uv,torch.tensor([4.8,4.8]),slopes,views[0],k,2,6)
    assert f.all() and (s==0).all() and valid.all()


def test_transport_does_not_require_target_slope_and_uses_only_four_fit_queries():
    images,views,k,maps=fixture(); features,texture=patch_features(images,5)
    uv=torch.tensor([[32.,30.],[29.,35.]]); old=torch.tensor([4.8,4.8])
    out,d=propose(uv,old,torch.full((2,2),float('nan')),maps,images,features,texture,views,k,0,list(range(1,7)),2,6,True)
    assert d['accepted'].all() and torch.allclose(out,torch.full_like(out,4),atol=1e-5)
    assert d['source_uv'].shape==(4,2,2) and d['incumbent_fallback'].all()
    maps['admitted'][:5]=False # Only held sources remain admitted; none can propose.
    same,empty=propose(uv,old,torch.zeros(2,2),maps,images,features,texture,views,k,0,list(range(1,7)),2,6,True)
    assert torch.equal(same,old) and not empty['eligible'].any()


def test_source_maps_use_original_depth_and_both_parent_admission_flags(monkeypatch):
    import variants.staged_reconstruction.plane_transport.inputs as module
    def parent(case,i):
        return {'uv':np.array([[2,2],[1,1]]),'original':np.array([4.,4.]),'updated':np.array([5.,5.]),
            'coefficients':np.array([[0,0,.25],[0,0,.25]]),'slope_admitted':np.array([True,True]),'eligible':np.array([True,False])}
    monkeypatch.setattr(module,'load_parent',parent)
    maps=module.make_maps('fixture',4,4,'cpu')
    assert maps['original'][0,2,2]==4 and maps['admitted'][0,2,2] and not maps['admitted'][0,1,1]


def test_identical_plane_cost_tie_keeps_first_source_and_no_held_improvement():
    images,views,k,maps=fixture(shift=0.)
    features,texture=patch_features(images,5); uv=torch.tensor([[32.,30.]])
    old=torch.tensor([4.8])
    out,d=propose(uv,old,torch.zeros(1,2),maps,images,features,texture,views,k,0,list(range(1,7)),2,6,True)
    assert d['selected_source'].item()==1 and torch.equal(out,old) and not d['accepted'].any()
