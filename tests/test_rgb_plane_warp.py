import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from variants.staged_reconstruction.rgb_plane_warp.core import fit_slopes,plane_support,warped_cost,propose
from variants.staged_reconstruction.adaptive_static.geometry import patch_features

P=json.loads((Path(__file__).resolve().parents[1]/'experiments/staged_reconstruction/S1_rgb_plane_warp/protocol.json').read_text())


def fixture(slope=0.,shift=8/35):
    k=torch.tensor([[35.,0,31.5],[0,35.,31.5],[0,0,1.]])
    yy,xx=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
    rx,ry=(xx-31.5)/35,(yy-31.5)/35
    views=torch.eye(4).repeat(7,1,1); views[:,0,3]=torch.tensor([0,shift,2*shift,3*shift,-shift,-2*shift,-3*shift])
    images=[]; depths=[]
    for view in views:
        ox=-view[0,3]; z=(4-slope*ox)/(1+slope*rx)
        wx,wy=ox+rx*z,ry*z
        g=.45+.15*torch.sin(wx*7)+.1*torch.cos(wy*9+wx*3)
        images.append(torch.stack([g,.9*g,.8*g])); depths.append(z)
    return torch.stack(images),views,k,torch.stack(depths)


def test_flags_off_identity_requires_no_inputs():
    old=torch.tensor([4.]); result,diag=propose(None,old)
    assert result is old and not diag['accepted'].any()


def test_cli_is_opt_in():
    r=subprocess.run([sys.executable,'-m','variants.staged_reconstruction.rgb_plane_warp.infer'],capture_output=True,text=True)
    assert r.returncode!=0 and 'opt-in' in r.stderr


def test_slope_exact_inverse_plane_and_parity_count():
    y,x=np.mgrid[:16,:16]; depth=1/(.003*(x-8)-.002*(y-8)+.25)
    out=fit_slopes(depth,np.array([[8,8]]),P)
    assert out['slope_admitted'].all() and np.allclose(out['coefficients'][0],[.003,-.002,.25])
    assert out['fit_neighbors'][0]==12 and out['held_neighbors'][0]==12


def test_slope_center_is_excluded_from_fit_but_checked():
    depth=np.full((16,16),4.); depth[8,8]=4.1
    out=fit_slopes(depth,np.array([[8,8]]),P)
    assert np.allclose(out['coefficients'][0],[0,0,.25]) and not out['slope_admitted'].any()


def test_bad_held_neighbor_rejects_even_perfect_fit():
    depth=np.full((16,16),4.); depth[8,9]=4.1
    out=fit_slopes(depth,np.array([[8,8]]),P)
    assert not out['slope_admitted'].any() and out['max_depth_residual'][0]>.09


def test_missing_degenerate_or_border_abstains():
    depth=np.zeros((16,16)); depth[8,:]=4
    assert not fit_slopes(depth,np.array([[8,8],[1,1]]),P)['slope_admitted'].any()
    depth[:]=4; depth[6:11:2,6:11:2]=np.nan
    assert not fit_slopes(depth,np.array([[8,8]]),P)['slope_admitted'].any()


def test_plane_support_represents_slanted_world_plane():
    _,views,k,depths=fixture(.4)
    uv=torch.tensor([[32.,30.],[29.,35.]])
    z=depths[0,uv[:,1].long(),uv[:,0].long()]
    slopes=torch.tensor([[.4/(4*35),0.]]).repeat(2,1)
    points,valid=plane_support(uv,z[None],slopes,views[0],k,2,6)
    assert valid.all() and torch.allclose(.4*points[...,0]+points[...,2],torch.full_like(points[...,2],4),atol=1e-6)


def test_invalid_plane_returns_unit_cost_without_nan():
    images,views,k,_=fixture(); features,texture=patch_features(images)
    uv=torch.tensor([[32.,30.]]); ref=features[0,:,30,32,None]
    pts,valid=plane_support(uv,torch.tensor([[4.]]),torch.tensor([[1.,0.]]),views[0],k,2,6)
    c=warped_cost(pts,valid,ref,texture[0,30,32,None],images[1],views[1],k)
    assert not valid.any() and c.item()==1


def test_raw_warp_same_camera_reproduces_reference_patch():
    images,views,k,_=fixture(); features,texture=patch_features(images)
    uv=torch.tensor([[32.,30.]])
    pts,valid=plane_support(uv,torch.tensor([[4.]]),torch.zeros(1,2),views[0],k,2,6)
    c=warped_cost(pts,valid,features[0,:,30,32,None],texture[0,30,32,None],images[0],views[0],k)
    assert c.item()<1e-6


def test_warp_gradients_into_depth_slope_and_raw_rgb():
    images,views,k,_=fixture(0.,.2); features,texture=patch_features(images)
    z=torch.tensor([[4.1]],requires_grad=True); slope=torch.tensor([[.001,.001]],requires_grad=True)
    source=images[2].clone().requires_grad_(True)
    pts,valid=plane_support(torch.tensor([[32.,30.]]),z,slope,views[0],k,2,6)
    c=warped_cost(pts,valid,features[0,:,30,32,None],texture[0,30,32,None],source,views[2],k)
    assert c.item()<1
    grads=torch.autograd.grad(c.sum(),[z,slope,source])
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)


def test_integer_textured_plane_proposal_uses_new_cost_for_both_depths():
    images,views,k,_=fixture(); features,texture=patch_features(images)
    uv=torch.tensor([[32.,30.],[29.,35.]])
    old=torch.tensor([4.8,4.8])
    result,d=propose(uv,old,torch.zeros(2,2),torch.ones(2,dtype=torch.bool),images,features,texture,views,k,0,list(range(1,7)),2,6,True)
    assert d['accepted'].all() and (result-4).abs().max()<.03
    assert (d['fit_new']<=d['fit_old']+1e-6).all() and (d['held_new']<=d['held_old']+1e-6).all()


def test_no_slope_admission_needs_no_rgb_access():
    old=torch.tensor([4.]); out,d=propose(None,old,admitted=torch.tensor([False]),sources=list(range(6)),enabled=True)
    assert torch.equal(out,old) and not d['accepted'].any()


def test_texture_or_chroma_invalid_returns_one():
    images,views,k,_=fixture(); f,t=patch_features(images)
    pts,v=plane_support(torch.tensor([[32.,30.]]),torch.tensor([[4.]]),torch.zeros(1,2),views[0],k,2,6)
    blue=images[0].clone(); blue[:2]=0
    for source in [torch.zeros_like(images[0]),torch.ones_like(images[0])*.4,blue]:
        c=warped_cost(pts,v,f[0,:,30,32,None],t[0,30,32,None],source,views[0],k)
        assert c.item()==1


def test_source_margin_is_required_for_every_support_sample():
    images,views,k,_=fixture(); f,t=patch_features(images)
    uv=torch.tensor([[3.,30.]])
    pts,v=plane_support(uv,torch.tensor([[4.]]),torch.zeros(1,2),views[0],k,2,6)
    c=warped_cost(pts,v,f[0,:,30,3,None],t[0,30,3,None],images[0],views[0],k)
    assert c.item()==1


def test_admitted_slope_with_invalid_current_plane_keeps_old_without_rgb():
    old=torch.tensor([4.]); uv=torch.tensor([[32.,30.]])
    out,d=propose(uv,old,torch.tensor([[1.,0.]]),torch.tensor([True]),
        views=torch.eye(4)[None],k=torch.eye(3),index=0,sources=list(range(6)),near=2,far=6,enabled=True)
    assert torch.equal(out,old) and not d['eligible'].any()
