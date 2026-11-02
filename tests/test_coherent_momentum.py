import json
import pytest
import torch
from variants.coherent_momentum.operator import polynomial,divergence,kinematics,momentum


def cloud():
    gen=torch.Generator().manual_seed(41)
    x=torch.randn(65,3,generator=gen,dtype=torch.double)
    return x,torch.tensor([0]),torch.arange(1,65)[None]


def test_quadratic_gradient_hessian_exact_and_units():
    x,anchor,neighbor=cloud();x=3.7*x+torch.tensor([5.,2.,-1.])
    y=torch.stack([x[:,0]**2+3*x[:,1]*x[:,2],2*x[:,1]+x[:,2]**2],-1)
    g,h,valid,_=polynomial(x,y,anchor,neighbor,ridge=0)
    assert valid.all()
    expected=torch.tensor([[2*x[0,0],3*x[0,2],3*x[0,1]],[0,2,2*x[0,2]]])
    assert torch.allclose(g[0],expected,atol=1e-10)
    assert h[0,0,0,0]==pytest.approx(2.) and h[0,0,1,2]==pytest.approx(3.)
    assert h[0,1,2,2]==pytest.approx(2.)


def test_affine_translation_and_rigid_rotation_have_zero_rate_divergence():
    x,a,n=cloud();zero=x.new_zeros(len(x),6)
    rotation=x.new_tensor([[0.,-2.,1.],[2.,0.,-3.],[-1.,3.,0.]])
    for v in [torch.ones_like(x)*3,x@rotation.T+2]:
        force,valid,_=divergence(x,v,zero,1.,a,n,ridge=0)
        assert valid.all() and force.abs().max()<1e-10


def test_quadratic_rate_and_linear_stress_divergence_analytic():
    x,a,n=cloud();v=torch.stack([x[:,0]**2,x[:,1]**2,x[:,2]**2],-1)
    s=x.new_zeros(len(x),6);s[:,:3]=x
    force,valid,parts=divergence(x,v,s,.6,a,n,ridge=0)
    # div dev sym grad(v) = 4/3 in each component; div(diag(x,y,z))=1.
    assert valid.all() and torch.allclose(force,torch.full_like(force,1.8),atol=1e-10)
    assert torch.allclose(parts['rate'],torch.full_like(force,.8),atol=1e-10)


@pytest.mark.parametrize('t',[0.,.4,1.])
def test_same_interval_trajectory_velocity_acceleration_and_rollout(t):
    q=torch.tensor(1.3,dtype=torch.double,requires_grad=True); h=1/199
    f=lambda s:torch.stack([q*s*s+2*s+3,q*s*s-4*s+1,2*q*s*s]).reshape(1,3)
    x,v,a,xm,xp,times=kinematics(f,t,h)
    assert 0<=times[0]<times[1]<times[2]<=1
    assert torch.allclose(a,torch.stack([2*q,2*q,4*q])[None],atol=1e-9)
    assert torch.allclose(xp,2*x-xm+h*h*a,atol=1e-12)
    assert torch.allclose(v,torch.stack([2*q*times[1]+2,2*q*times[1]-4,4*q*times[1]])[None])
    a.square().sum().backward();assert q.grad.isfinite() and q.grad.abs()>0


def test_manufactured_momentum_balance_and_gradients():
    x,a,n=cloud();q=torch.tensor(.6,dtype=torch.double,requires_grad=True)
    v=torch.stack([x[:,0]**2,x[:,1]**2,x[:,2]**2],-1)
    stress=torch.zeros(len(x),6,dtype=torch.double,requires_grad=True)
    acceleration=torch.full_like(x,.8)
    r,valid,_=momentum(x,v,stress,acceleration,q,a,n,ridge=0)
    assert valid.all() and r.abs().max()<1e-10
    (r-1).square().sum().backward()
    assert stress.grad.isfinite().all() and stress.grad.abs().sum()>0 and q.grad.abs()>0


def test_coplanar_neighborhood_is_rejected():
    x,a,n=cloud();x[:,2]=0
    _,_,valid,ratio=polynomial(x,x,a,n)
    assert not valid.any() and ratio.max()<1e-6


def test_new_flag_off_returns_exact_m1_path(tmp_path,monkeypatch):
    from variants.coherent_momentum.runtime import Tuning,Parent
    path=tmp_path/'config.json';path.write_text(json.dumps({'enabled':True,'coherent_momentum':False}))
    tune=Tuning(path,tmp_path);value=torch.tensor(2.,requires_grad=True)
    monkeypatch.setattr(Parent,'momentum_loss',lambda *args:value)
    assert tune.momentum_loss(100,None,None,None) is value
