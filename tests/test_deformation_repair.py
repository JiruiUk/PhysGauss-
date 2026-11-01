import json
import random
import pytest
import torch
from variants.deformation_repair.model import motion_gate,rendered_covariance,covariance_curvature,time_stencil
from variants.coverage_tuning.model import local_affine_loss


def test_motion_gate_rejects_independent_touching_motion_but_retains_common_translation():
    x=torch.tensor([[0.,0,0],[.001,0,0],[.02,0,0]])
    a=torch.tensor([0]);n=torch.tensor([[1,2]]);v=torch.ones(1,2,dtype=torch.bool)
    assert motion_gate(x,[x,x+1,x+2],a,n,v).all()
    y=x.clone();y[1,0]+=1
    gate=motion_gate(x,[x,y,y],a,n,v)
    assert not gate[0,0] and gate[0,1]


def test_motion_gate_retains_neighboring_points_under_rigid_quarter_turn():
    x=torch.tensor([[0.,0,0],[.01,0,0],[0,.01,0]])
    y=torch.stack((-x[:,1],x[:,0],x[:,2]),-1)
    assert motion_gate(x,[x,y],torch.tensor([0]),torch.tensor([[1,2]]),torch.ones(1,2,dtype=torch.bool)).all()


def test_elastic_affine_is_free_but_local_tear_has_finite_gradients():
    x=torch.tensor([[0.,0,0],[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]],requires_grad=True)
    a=torch.tensor([0]);n=torch.arange(1,9)[None];v=torch.ones(1,8,dtype=torch.bool)
    affine=(x.detach()*torch.tensor([2.,.5,1.2])+3).requires_grad_()
    assert local_affine_loss(x,affine,a,n,v)<1e-10
    torn=affine.detach().clone();torn[-1,0]+=.5;torn.requires_grad_()
    loss=local_affine_loss(x,torn,a,n,v);loss.backward()
    assert loss>0 and torch.isfinite(torn.grad).all() and torn.grad.abs().sum()>0 and x.grad is None


def test_covariance_is_psd_sign_invariant_and_has_correct_identity_scale():
    scale=torch.tensor([[.5,1.,2.]])
    q=torch.tensor([[1.,0,0,0]])
    assert torch.equal(rendered_covariance(scale,q),torch.diag_embed(scale.square()))
    raw=torch.tensor([[1.3,.2,.4,-.1]])
    a=rendered_covariance(scale,raw)
    assert torch.allclose(a,a.transpose(-1,-2)) and torch.linalg.eigvalsh(a).min()>=-1e-7
    assert torch.equal(a,rendered_covariance(-scale,-raw))


def test_covariance_prior_leaves_constant_shape_and_log_linear_size_free():
    a=torch.eye(3)[None]
    loss,_=covariance_curvature(a,a*2,a*4)
    assert loss==0


def test_covariance_spike_is_penalized_and_gradients_reach_scale_and_rotation():
    s=torch.tensor([[.7,.9,1.2]],requires_grad=True);q=torch.tensor([[1.2,.1,.2,.3]],requires_grad=True)
    a=torch.eye(3)[None];b=rendered_covariance(s,q)
    loss,_=covariance_curvature(a,b,a);loss.backward()
    assert loss>0 and all(torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in [s,q])


def test_time_stencils_use_unchanged_normalized_clock_without_extrapolation():
    for t in [0.,1/199,.5,1.]:
        a,b,c=time_stencil(t,1/199)
        assert 0<=a<b<c<=1 and b-a==pytest.approx(c-b)
    with pytest.raises(ValueError):time_stencil(-1,.01)


def test_zero_new_weights_returns_parent_object_without_extra_rng_or_scene_access(tmp_path,monkeypatch):
    from variants.deformation_repair.runtime import Tuning,Parent
    p=tmp_path/'config.json';p.write_text(json.dumps({'enabled':True,'integrity_enabled':True}))
    expected=object();monkeypatch.setattr(Parent,'coverage_loss',lambda *args:expected)
    tuning=Tuning(p,tmp_path);state=random.getstate()
    assert tuning.coverage_loss(9000,object(),object(),object(),object()) is expected and state==random.getstate()
