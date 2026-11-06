import math
import numpy as np
import torch
from variants.staged_reconstruction.i10_ordinal_depth.core import ranking,sampled_depth,transition,make_pairs

P=dict(weight=.001,margin_ratio=1.02,log_error_scale=.05,minimum_pixel_separation=8,raw_normalized_gap_min=.1,maximum_prior_boundary_jump=.1,maximum_pairs_per_frame=4)


def test_default_off_identity():
    x=object();assert ranking(x,enabled=False)==(x,None)


def test_correct_order_zero_and_violated_order_gradient():
    base=torch.tensor(1.);z=torch.tensor([[1.,2.],[2.,1.]],requires_grad=True)
    total,aux=ranking(base,z,torch.ones(2),torch.ones(2),torch.ones(2,dtype=torch.bool),P,True)
    assert aux>0;total.backward();assert z.grad[0].abs().sum()==0 and z.grad[1,0]>0 and z.grad[1,1]<0


def test_absolute_scale_does_not_change_ordinal_loss():
    z=torch.tensor([[2.,1.]],requires_grad=True)
    f=lambda x:ranking(x.sum()*0,x,torch.ones(1),torch.ones(1),torch.ones(1,dtype=torch.bool),P,True)[1]
    assert torch.allclose(f(z),f(z*7),atol=1e-9)


def test_confidence_detached_and_fixed_missing_denominator():
    z=torch.tensor([[2.,1.],[2.,1.]],requires_grad=True);c=torch.ones(2,requires_grad=True);s=torch.ones(2,requires_grad=True)
    _,a=ranking(z.sum()*0,z,s,c,torch.tensor([True,False]),P,True)
    _,b=ranking(z.sum()*0,z,s,c,torch.ones(2,dtype=torch.bool),P,True)
    assert torch.allclose(a*2,b);a.backward();assert c.grad is None and s.grad is None and z.grad.isfinite().all()


def test_sample_native_D_over_alpha_and_gradient():
    d=torch.full((1,8,8),2.,requires_grad=True);a=torch.full((1,8,8),.5,requires_grad=True)
    z,alpha=sampled_depth(dict(depth=d,alpha=a),[[3.5,3.5],[4.,4.]])
    assert torch.allclose(z,torch.ones(2)*4);z.sum().backward();assert d.grad.isfinite().all() and a.grad.isfinite().all()


def test_unknown_pause_revoke_restore():
    s=transition(None,True,True,True);assert s['status']=='active'
    s=transition(s,False,False);assert s['status']=='suspended'
    s=transition(transition(s,True,False),True,False);assert s['status']=='revoked'
    s=transition(transition(s,True,True),True,True);assert s['status']=='active'


def fixture():
    y,x=np.mgrid[:64,:96];raw=(1+x*.01+y*.001).astype(np.float32)
    uv=[[4,4],[90,4],[90,60],[4,60]];anchors=[dict(id=str(i),group=str(i),split='fit0',uv=u,depth=1.) for i,u in enumerate(uv)]
    c=dict(passed=True,uv=uv,fit_ids=[0,1,2,3],coefficients=[1.,0.],split_coefficients=[[1.,0.],[1.,0.]])
    track=dict(passed=True,id='track',row=5,start=0,uv=[[30.,30.]]*12,fit_ids=[0,2,4,6,8,10],held_ids=[1,3,5,7,9,11],motion_P90_relative=.01)
    a=dict(candidate=dict(motion=[track],calibration=[c]*12),normalizations=[(1.,1.)]*12,anchor_bank=[anchors]*12)
    return a,[raw]*12


def test_pair_building_fit_held_separation_and_replay():
    a,raw=fixture();x=make_pairs(a,raw,P);assert x==make_pairs(a,raw,P) and len(x['pairs'])>0
    assert all(q['frame']%2==0 and not q['auxiliary_held'] for q in x['pairs'])
    assert make_pairs(a,raw,P,True)['pairs']==[]


def test_weak_or_overlapping_order_unknown():
    a,raw=fixture();assert make_pairs(a,[np.ones_like(raw[0])]*12,P)['pairs']==[]
    for c in a['candidate']['calibration']:c['split_coefficients']=[[1.,0.],[1.,10.]]
    assert make_pairs(a,raw,P)['pairs']==[]


def test_unadmitted_motion_has_no_targets():
    a,raw=fixture();a['candidate']['motion'][0]['passed']=False;assert make_pairs(a,raw,P)['pairs']==[]


def test_empty_loss_no_nan():
    z=torch.empty((0,2),requires_grad=True);base=torch.tensor(1.)
    total,aux=ranking(base,z,torch.empty(0),torch.empty(0),torch.empty(0,dtype=torch.bool),P,True)
    assert total==base and aux==0
