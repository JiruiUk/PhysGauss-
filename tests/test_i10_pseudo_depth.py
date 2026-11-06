import numpy as np
import pytest
import torch
from variants.staged_reconstruction.i10_pseudo_depth.core import (in_hull,depth_at,calibrate,boundary,backproject,motion_check,weak_depth,transition)

P=dict(deduplicate_px=4,minimum_fit=6,minimum_check=4,minimum_split_fit=3,minimum_extent=.1,minimum_inverse_spread=.01,relative_median_max=.05,relative_P90_max=.1)
M=dict(minimum_valid_fit=4,minimum_valid_check=4,motion_median_max=.02,motion_P90_max=.05)

def fixture():
    y,x=np.mgrid[:64,:96];raw=(1+x*.01+y*.003).astype(np.float32)
    uv=np.array([(x,y) for y in range(4,61,8) for x in range(4,93,8)],float)
    anchors=[dict(id=str(i),uv=q.tolist(),depth=float(1/(.2*raw[int(q[1]),int(q[0])]+.1))) for i,q in enumerate(uv)]
    return raw,uv,anchors

def test_exact_affine():
    raw,uv,a=fixture();q=calibrate(raw,a,P)
    assert q['passed'] and np.allclose(q['coefficients'],[.2,.1],atol=1e-6)
    assert q['split_P90_relative']<1e-5

def test_bad_held_anchors_cannot_refit_coefficients():
    raw,uv,a=fixture();before=calibrate(raw,a,P)
    for q in a:
        x,y=q['uv']
        if (x//8+y//8)%3==0:q['depth']*=1.5
    after=calibrate(raw,a,P)
    assert not after['passed'] and after['coefficients']==before['coefficients']

def test_missing_calibration_unknown():
    raw,uv,a=fixture();q=calibrate(raw,a[:2],P)
    assert not q['passed'] and q['coefficients'] is None

def test_negative_or_constant_prior_rejected():
    raw,uv,a=fixture()
    assert not calibrate(-raw,a,P)['passed']
    assert not calibrate(np.ones_like(raw),a,P)['passed']

def test_spatial_duplicates_do_not_inflate_support():
    raw,uv,a=fixture();q=calibrate(raw,a+a,P)
    assert q['anchors']==len(a)
    assert not(set(q['fit_ids'])&set(q['check_ids']))

def test_hull_no_extrapolation_or_degenerate():
    assert in_hull([[.5,.5],[2,2]],[[0,0],[1,0],[0,1],[1,1]]).tolist()==[True,False]
    assert not in_hull([[1,1]],[[0,0],[1,1],[2,2]]).any()

def test_invalid_depth_no_clamping():
    assert np.isnan(depth_at(np.ones((8,8)),[[3,3]],[-1,0])).all()

def test_boundary_and_off_image_unknown():
    raw=np.ones((16,16),np.float32);raw[:,8:]=2
    c,good=boundary(raw,[[3,3],[7,7],[0,0]],[1,0])
    assert good.tolist()==[True,True,False] and c[0]==0 and c[1]>.1 and np.isnan(c[2])

def test_camera_motion_is_removed():
    v=np.repeat(np.eye(4)[None],12,0);v[:,0,3]=np.arange(12)*.1
    uv=np.c_[np.arange(12)*.1,np.zeros(12)]
    assert np.allclose(backproject(uv,np.ones(12),v,np.eye(3)),[0,0,1])

def test_general_camera_rotation_translation_roundtrip():
    from variants.staged_reconstruction.i10_tracking.core import project_world
    angle=.4;r=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    v=np.repeat(np.eye(4)[None],12,0);v[:,:3,:3]=r;v[:,:3,3]=[.2,-.1,.3]
    points=np.c_[np.arange(12)*.03,np.zeros(12),np.ones(12)*2]
    k=np.array([[100,0,80],[0,100,45],[0,0,1]])
    uv,z=project_world(points[:,None],v,k)
    assert np.allclose(backproject(uv[:,0],z[:,0],v,k),points)

def test_prior_disabled_identity():
    from variants.staged_reconstruction.pretrained_depth.model import predict,load_model
    original=object();assert predict(original,enabled=False) is original
    assert load_model(None,None,False) is None

def test_motion_compensation_and_fit_held_separation():
    parent=np.c_[np.arange(12)*.05,np.zeros(12),np.ones(12)]
    prior=parent+np.array([.01,.02,.03]);q=motion_check(prior,parent,np.ones(12,bool),np.ones(12),M)
    assert q['passed'] and q['motion_median_relative']<1e-8 and q['camera_only_median_relative']>.02
    prior[1::2,1]+=.2;q=motion_check(prior,parent,np.ones(12,bool),np.ones(12),M)
    assert not q['passed'] and np.allclose(q['fit_offset'],[.01,.02,.03])

def test_missing_motion_remains_unknown():
    p=np.ones((12,3));q=motion_check(p,p,np.zeros(12,bool),np.ones(12),M)
    assert not q['passed'] and q['reason']=='insufficient_observations'

def test_flags_off_exact_identity():
    base=object();assert weak_depth(base,enabled=False) is base

def test_weak_depth_gradient_confidence_detached():
    z=torch.tensor([1.,2.],requires_grad=True);target=torch.tensor([1.1,1.9],requires_grad=True);c=torch.ones(2,requires_grad=True)
    loss=weak_depth(z.sum()*0,z,target,c,True);loss.backward()
    assert z.grad.isfinite().all() and z.grad.abs().sum()>0 and target.grad is None and c.grad is None

def test_weak_depth_empty_and_invalid():
    z=torch.empty(0,requires_grad=True);base=torch.tensor(2.)
    assert weak_depth(base,z,z,z,True)==base
    with pytest.raises(AssertionError):weak_depth(base,torch.ones(1),torch.zeros(1),torch.ones(1),True)

def test_suspend_revoke_restore():
    a=transition(None,'pass');a=transition(a,'pass');assert a['status']=='active'
    b=transition(a,'unknown');assert b['status']=='suspended' and b['failures']==0
    b=transition(transition(b,'contradiction'),'contradiction');assert b['status']=='revoked'
    b=transition(transition(b,'pass'),'pass');assert b['status']=='active'
