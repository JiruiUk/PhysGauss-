import json
import numpy as np
import torch
from variants.transfer_integrity.model import static_position, static_anchor, static_covariance, local_shape, reprojection, fixed_edges


def test_static_pair_has_finite_gradients_into_both_times():
    a=torch.tensor([[0.,0,1]],requires_grad=True);b=torch.tensor([[.2,0,1]],requires_grad=True)
    value=static_position(a,b);value.backward()
    assert value>0 and a.grad[0,0]<0 and b.grad[0,0]>0


def test_static_covariance_is_matrix_based_and_zero_when_unchanged():
    a=torch.eye(3)[None].requires_grad_()
    assert static_covariance(a,a.clone())==0
    b=(a.detach()*2).requires_grad_();static_covariance(b,a.detach()).backward()
    assert torch.isfinite(b.grad).all() and b.grad.abs().sum()>0


def test_anchor_targets_are_detached_and_tolerance_allows_surface_offset():
    target=torch.zeros(1,3,requires_grad=True)
    p=torch.tensor([[.01,0,0]],requires_grad=True)
    assert static_anchor(p,target)==0
    p=torch.tensor([[.2,0,0]],requires_grad=True);static_anchor(p,target).backward()
    assert p.grad[0,0]>0 and target.grad is None


def test_local_shape_allows_rigid_motion_and_moderate_elastic_strain():
    a=torch.tensor([[0.,0,0]]);b=torch.tensor([[1.,0,0]]);length=torch.ones(1)
    assert local_shape(a+7,torch.tensor([[0.,1.,0]])+7,length)==0
    assert local_shape(a,b*1.5,length)==0


def test_local_shape_penalizes_tearing_and_collapse_with_restoring_gradient():
    a=torch.zeros(1,3)
    for x,sign in [(4.,1),(.1,-1)]:
        b=torch.tensor([[x,0,0]],requires_grad=True)
        value=local_shape(a,b,torch.ones(1));value.backward()
        assert value>0 and sign*b.grad[0,0]>0 and torch.isfinite(b.grad).all()


def test_edges_never_connect_unknown_or_different_objects():
    p=np.array([[0.,0,0],[.01,0,0],[.011,0,0],[.02,0,0],[.03,0,0]])
    e=fixed_edges(p,np.arange(5),np.array([0,0,1,1,-1]))
    assert len(e)>0 and all(int(a)//2==int(b)//2 and a<4 and b<4 for a,b,_ in e)


def test_reprojection_uses_true_pixel_target_and_cannot_hide_behind_camera():
    view=torch.eye(4);k=torch.tensor([[100.,0,200],[0,100.,100],[0,0,1.]])
    target=torch.tensor([[200.,100.]])
    assert reprojection(torch.tensor([[0.,0,2.]]),view,k,target)[0]==0
    p=torch.tensor([[.2,0,2.]],requires_grad=True);loss,epe=reprojection(p,view,k,target)
    loss.backward();assert epe==10 and p.grad[0,0]>0
    assert reprojection(torch.tensor([[0.,0,-1.]]),view,k,target)[0]>0


def test_flags_off_returns_exact_parent_loss_without_data(tmp_path,monkeypatch):
    from variants.transfer_integrity.runtime import Tuning,Parent
    path=tmp_path/'config.json';path.write_text(json.dumps({'enabled':True,'transfer_integrity_enabled':False}))
    t=Tuning(path,tmp_path);sentinel=object()
    monkeypatch.setattr(Parent,'coverage_loss',lambda *args:sentinel)
    assert t.coverage_loss(50010,None,None,None,None) is sentinel
    assert not hasattr(t,'binding')


def test_original_image_and_physics_paths_inherited_unchanged():
    from variants.transfer_integrity.runtime import Tuning,Parent
    for method in ['image_loss','physics_time','time_step','regularization','kinematic_loss','after_update','evaluate']:
        assert getattr(Tuning,method) is getattr(Parent,method)
