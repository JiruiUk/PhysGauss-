import numpy as np
import torch
from types import SimpleNamespace
from variants.staged_reconstruction.i10_track_train.core import observations,add_tracks


def test_only_accepted_persistent_rows_and_duplicate_mean():
    accepted=dict(start=2,uv=[[1,2],[3,4]],binding=dict(accepted=True,row=7))
    duplicate=dict(start=2,uv=[[3,4],[5,6]],binding=dict(accepted=True,row=7))
    rejected=dict(start=2,uv=[[99,99]],binding=dict(accepted=False,row=9))
    result=observations(dict(rows=[accepted,duplicate,rejected]))
    assert result=={2:[(7,[2.,3.])],3:[(7,[4.,5.])]}


def test_no_accepted_labels_no_fake_targets():assert observations(dict(rows=[dict(start=0)]))=={}


def test_flags_off_no_scene_access_returns_rgb_identity():
    rgb=torch.tensor(2.,requires_grad=True)
    loss,term=add_tracks(rgb,None,None,None,None)
    assert loss is rgb and term.item()==0


def test_empty_observation_no_scene_access():
    rgb=torch.tensor(2.);loss,term=add_tracks(rgb,None,None,None,[],True)
    assert loss is rgb and term.item()==0


def test_only_bound_3d_rows_receive_direct_track_gradient():
    xyz=torch.tensor([[.1,.2,4.],[.2,.1,4.]],requires_grad=True)
    scene=SimpleNamespace(dynamic_at=lambda t:SimpleNamespace(get_xyz=xyz))
    cam=SimpleNamespace(FoVx=np.pi/2,world_view_transform=torch.eye(4))
    loss,term=add_tracks(torch.tensor(0.),scene,cam,.5,[(1,[220,120])],True)
    loss.backward();assert xyz.grad[0].abs().sum()==0
    assert torch.isfinite(xyz.grad[1]).all() and xyz.grad[1].abs().sum()>0 and term>0
