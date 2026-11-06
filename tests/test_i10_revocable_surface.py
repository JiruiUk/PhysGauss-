from copy import deepcopy
import numpy as np
import torch
from variants.staged_reconstruction.i10_revocable_surface.core import (
    build_patches,build_edges,geometry,patch_measurement,transition,edge_measurement,add_constraints)
from variants.staged_reconstruction.i10_tracking.io import ROOT,read

P=read(ROOT/'experiments/staged_reconstruction/RubberDuck_I10_revocable_surface/protocol.json')


def passed():
    return dict(kind='pass',passed=True,confidence=.8,frames=[dict(passed=True)]*12)


def state():return transition(None,passed(),P,True)


def patch(id='a',row=0,start=0,x=0.):
    return dict(id=id,start=start,rows=[row],weights=[1.],uv=[[199.5,112.]]*12,parent_centers=[[x,0,1.]]*12)


def test_first_pass_can_initialize_without_learning_confidence():
    assert state()['status']=='active' and state()['confidence']==.8


def test_unknown_suspends_without_deleting_or_claiming_background():
    old=state();new=transition(old,dict(kind='unknown',confidence=0.),P)
    assert new['status']=='suspended' and new['confidence']==0 and old['status']=='active'


def test_two_contradictions_revoke_then_two_passes_restore():
    bad=dict(kind='contradiction',confidence=0.)
    a=transition(state(),bad,P);b=transition(a,bad,P)
    assert a['status']=='suspended' and b['status']=='revoked'
    c=transition(b,passed(),P);d=transition(c,passed(),P)
    assert c['status']=='revoked' and c['confidence']==0 and d['status']=='active'


def test_unknown_interrupts_consecutive_conflict_count():
    bad=dict(kind='contradiction',confidence=0.)
    a=transition(state(),bad,P);a=transition(a,dict(kind='unknown'),P);a=transition(a,bad,P)
    assert a['status']=='suspended' and a['fail_streak']==1


def test_state_deepcopy_roundtrip_no_alias():
    import json
    a=state();b=json.loads(json.dumps(a));assert a==b
    b['measurement']['frames'][0]['passed']=False;assert a['measurement']['frames'][0]['passed']


def test_fit_only_patch_members_and_weights():
    xyz=np.zeros((12,3,3));xyz[:,:,2]=1;xyz[:,1,0]=.005;xyz[:,2,0]=.1
    projected=np.zeros((12,3,2));projected[:,1,0]=.2
    observations={str(i):[dict(passed=True,numerator=float(i+1)) for _ in range(12)] for i in range(3)}
    t=dict(accepted=True,id='a',start=0,band=0,uv=[[0,0]]*12,binding=dict(row=0,eligible_rows=[0,1,2]),visibility_by_row=observations)
    a=build_patches(dict(rows=[t]),xyz,projected,P)
    xyz[P['held'],1,0]=100;projected[P['held'],1,0]=100
    b=build_patches(dict(rows=[t]),xyz,projected,P)
    assert a[0]['rows']==b[0]['rows']==[0,1] and a[0]['weights']==b[0]['weights']==[1/3,2/3]


def test_graph_requires_distinct_independently_bound_nonoverlap_same_window():
    a=patch();b=patch('b',1,x=.02)
    assert len(build_edges([a,b],P))==1
    b['rows']=[0];assert not build_edges([a,b],P)
    b['rows']=[1];b['start']=4;assert not build_edges([a,b],P)


def test_held_can_reject_fit_proposed_edge():
    a=patch();b=patch('b',1,x=.02);b['parent_centers'][1]=[1.,0,1.]
    edges=build_edges([a,b],P);assert len(edges)==1
    lengths=np.linalg.norm(np.array(a['parent_centers'])-b['parent_centers'],axis=-1)
    assert edge_measurement(lengths,state(),state(),P)['kind']=='contradiction'


def test_geometry_accepts_rotation_not_large_stretch():
    assert geometry([.02]*12,P['fit'],P)
    assert not geometry([.01,.04]*6,list(range(12)),P)


def test_appearance_missing_is_unknown_not_zero_error():
    vis=[dict(known=True,passed=True,share=.1)]*12
    app=[dict(ncc=1.,observed_std=0.,rendered_std=0.)]*12
    m=patch_measurement(np.zeros((12,2)),np.ones(12),vis,app,np.zeros((12,2)),P)
    assert not m['passed'] and m['kind']=='unknown' and m['confidence']==0


def test_split_and_frame_checks_reject_confident_misprojection():
    vis=[dict(known=True,passed=True,share=.1)]*12;app=[dict(ncc=1.,observed_std=3.,rendered_std=3.)]*12
    uv=np.zeros((12,2));m=patch_measurement(uv,np.ones(12),vis,app,uv,P)
    assert m['passed'] and m['confidence']==1
    uv[1]=[10,0];m=patch_measurement(uv,np.ones(12),vis,app,np.zeros((12,2)),P)
    assert not m['passed'] and m['kind']=='contradiction'


def args(frame=0):
    patches=[patch(),patch('b',1,x=.02)];edges=build_edges(patches,P)
    w=torch.tensor([[0.,0.,1.],[.02,0,1.]],requires_grad=True)
    k=torch.tensor([[10.,0,199.5],[0,10.,112.],[0,0,1.]])
    return w,torch.eye(4),k,frame,patches,edges,[state(),state()],[state()],P


def test_default_off_is_exact_identity_without_touching_inputs():
    base=torch.tensor(2.,requires_grad=True)
    assert add_constraints(base,*([None]*9))[0] is base


def test_held_and_outside_frames_have_no_auxiliary_loss():
    for frame in [1,12,119]:
        w,*rest=args(frame);_,a,b,counts=add_constraints(w.sum()*0,w,*rest,enabled=True)
        assert a==b==0 and counts['tracks']==counts['edges']==0


def test_revocation_disables_track_and_attached_edge():
    w,v,k,f,patches,edges,states,edge_states,p=args()
    states[0]['status']='revoked'
    _,a,b,counts=add_constraints(w.sum()*0,w,v,k,f,patches,edges,states,edge_states,p,True)
    assert counts['tracks']==1 and counts['edges']==0 and b==0


def test_scalar_length_loss_rotation_invariant_and_gradient_finite():
    values=args();w=values[0]
    _,_,local,_=add_constraints(w.sum()*0,*values,True)
    assert abs(float(local.detach()))<1e-8
    rotated=w@torch.tensor([[0.,-1.,0],[1.,0,0],[0,0,1.]])
    _,_,other,_=add_constraints(w.sum()*0,rotated,*values[1:],True)
    assert torch.allclose(local,other)
    perturbed=w+torch.tensor([[0.,0,0],[.003,0,0]])
    _,track,local,_=add_constraints(w.sum()*0,perturbed,*values[1:],True)
    (track+local).backward();assert torch.isfinite(w.grad).all() and w.grad.abs().sum()>0


def test_projection_guard_suspends_stale_current_frame():
    values=args();w=values[0]+torch.tensor([[100.,0,0],[0.,0,0]])
    _,a,b,c=add_constraints(w.sum()*0,w,*values[1:],True)
    assert c['tracks']==1 and c['edges']==0
