import json
from pathlib import Path
import numpy as np
import pytest
import torch
from variants.staged_reconstruction.i10_tracking.core import (
    grayscale,seeds,qualify,project_world,stationary_test,bind_track,tracking_objective)

P=json.loads((Path(__file__).parents[1]/'experiments/staged_reconstruction/RubberDuck_I10_tracking/protocol.json').read_text())

def cameras():
    views=np.repeat(np.eye(4)[None],12,axis=0);views[:,0,3]=-np.linspace(-1,1,12)
    return views,np.array([[200.,0,199.5],[0,200,112],[0,0,1]])

def test_channel_permutation_and_full_image_seeds():
    rgb=np.random.default_rng(2).integers(0,256,(225,400,3),dtype=np.uint8)
    assert np.array_equal(grayscale(rgb),grayscale(rgb[...,::-1]))
    s=seeds(grayscale(rgb),P);assert s.shape==(96,2)
    assert np.array_equal(s,seeds(grayscale(rgb[...,::-1]),P))
    assert len(set((int(y)//57,int(x)//100) for x,y in s))>=12

def test_no_texture_no_seed():assert len(seeds(np.zeros((225,400)),P))==0

def test_forced_query_visibility_excluded_but_real_reverse_closure_required():
    gray=np.random.default_rng(4).normal(100,20,(225,400)).astype(np.float32)
    f=np.tile([[[40.,40.]]],(12,1,1));r=f.copy();v=np.ones((12,1),bool);rv=v.copy()
    v[0]=False;rv[-1]=False
    accepted,_=qualify(f,r,v,rv,[gray]*12,P);assert accepted[0]
    r[0,0,0]+=2
    accepted,_=qualify(f,r,v,rv,[gray]*12,P);assert not accepted[0]

def test_interior_reverse_mismatch_rejected():
    gray=np.random.default_rng(5).normal(100,20,(225,400)).astype(np.float32)
    f=np.tile([[[40.,40.]]],(12,1,1));r=f.copy();r[5,0,1]+=1
    accepted,_=qualify(f,r,np.ones((12,1),bool),np.ones((12,1),bool),[gray]*12,P)
    assert not accepted[0]

def test_stationary_camera_motion_compensated():
    v,k=cameras();uv,z=project_world(np.array([[.2,.3,4.]]),v,k)
    result=stationary_test(uv[:,0],v,k,P)
    assert result['label']=='stationary_compatible' and result['max_px']<1e-8

def test_moving_point_not_triangulated_as_gt():
    v,k=cameras();xyz=np.tile([[[0.,0.,4.]]],(12,1,1));xyz[:,0,1]=.25*np.sin(np.arange(12)*.7)
    uv,z=project_world(xyz,v,k);result=stationary_test(uv[:,0],v,k,P)
    assert result['label']=='motion_candidate'

def test_degenerate_views_remain_unknown():
    v,k=cameras();v[:]=v[0];uv,z=project_world(np.array([[0.,0.,4.]]),v,k)
    assert stationary_test(uv[:,0],v,k,P)['label']=='unknown'

def fixture_binding():
    v,k=cameras();xyz=np.tile(np.array([[[0.,0.,4.],[.2,0.,4.],[0.,0.,4.01]]]),(12,1,1))
    uv,z=project_world(xyz,v,k);return xyz,uv,z

def test_persistent_binding_not_perframe_nearest():
    xyz,uv,z=fixture_binding();result=bind_track(uv[:,0],xyz,uv,z,P)
    assert result['accepted'] and result['row']==0 and not result['world_depth_is_independent_GT']

def test_held_observations_cannot_select_id():
    xyz,uv,z=fixture_binding();target=uv[:,0].copy();target[1::2]=uv[1::2,1]
    result=bind_track(target,xyz,uv,z,P)
    assert result['row']==0 and not result['accepted'] and not result['checks']['held']

def test_behind_camera_not_bound():
    xyz,uv,z=fixture_binding();z[:]=-1
    assert not bind_track(uv[:,0],xyz,uv,z,P)['accepted']

def test_3d_ambiguity_rejects():
    xyz,uv,z=fixture_binding();xyz[:,1]=xyz[:,0]+[0,0,1];uv[:,1]=uv[:,0]
    assert not bind_track(uv[:,0],xyz,uv,z,P)['accepted']

def test_flags_off_identity_without_inputs_or_mutation():
    original=torch.tensor(2.,requires_grad=True)
    assert tracking_objective(original,None,None,None,None) is original

def test_3d_track_gradient_and_zero_at_match():
    v,k=cameras();world=torch.tensor(np.tile([[[.1,.2,4.]]],(12,1,1)),requires_grad=True)
    views=torch.tensor(v);intrinsic=torch.tensor(k);uv,_=project_world(world.detach().numpy(),v,k)
    loss=tracking_objective(torch.tensor(0.),world,views,intrinsic,torch.tensor(uv)+1,True)
    loss.backward();assert torch.isfinite(world.grad).all() and world.grad.abs().sum()>0
    exact=tracking_objective(torch.tensor(0.),world,views,intrinsic,torch.tensor(uv),True)
    assert abs(float(exact.detach()))<1e-12

def test_invalid_depth_not_used_as_supervision():
    v,k=cameras();world=torch.tensor(np.tile([[[0.,0.,-1.]]],(12,1,1)))
    with pytest.raises(AssertionError):tracking_objective(torch.tensor(0.),world,torch.tensor(v),torch.tensor(k),torch.zeros(12,1,2),True)
