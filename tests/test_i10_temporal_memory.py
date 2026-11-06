from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import torch
from variants.staged_reconstruction.i10_temporal_memory.core import (
    propose_links, build_bank, bind_memory, measurement, keyframes, add_memory)
from variants.staged_reconstruction.i10_revocable_surface.core import transition

P=json.loads((Path(__file__).resolve().parents[1]/'experiments/staged_reconstruction/RubberDuck_temporal_memory/protocol.json').read_text())

def track(start, offset=0., name=None):
    return dict(id=name or str(start),start=start,uv=[[100+f*.2+offset,80.] for f in range(start,start+12)])

def evidence():
    return dict(rows=[dict(id='0',accepted=True,binding=dict(row=0))])

def test_real_overlap_stitches_and_retains_every_observation():
    bank=build_bank([track(0),track(4),track(8)],evidence(),P)
    assert len(bank['memories'])==1
    m=bank['memories'][0];assert m['candidate'] and m['frames']==list(range(20))
    assert len(m['track_ids'])==3 and sum(m['overlap_observation_counts'])==36

def test_fit_nomination_does_not_read_held():
    tracks=[track(0),track(4)];a=propose_links(tracks,P)
    for t in tracks:
        for i,p in enumerate(t['uv']):
            if (t['start']+i)%2:p[0]+=100*(t['start']+1)
    assert propose_links(tracks,P)==a
    assert not build_bank(tracks,evidence(),P)['memories'][0]['candidate']

def test_gap_is_not_invented_or_reidentified():
    bank=build_bank([track(0),track(20)],evidence(),P)
    assert len(bank['memories'])==2 and not any(m['candidate'] for m in bank['memories'])

def test_ambiguous_nearby_surfaces_remain_separate():
    tracks=[track(0),track(4,0.,'a'),track(4,.1,'b')]
    bank=build_bank(tracks,evidence(),P)
    assert not bank['fit_links'] and len(bank['memories'])==3

def test_input_order_and_no_mutation():
    tracks=[track(0),track(4),track(8)];before=deepcopy(tracks)
    assert build_bank(tracks,evidence(),P)==build_bank(tracks[::-1],evidence(),P)
    assert tracks==before

def test_nonfinite_overlap_stays_unknown():
    t=track(4);t['uv'][0][0]=float('nan')
    assert not propose_links([track(0),t],P)[0]

def fixture():
    m=build_bank([track(0),track(4)],evidence(),P)['memories'][0]
    n=len(m['frames']);xyz=np.zeros((n,1,3));xyz[:,:,2]=1
    proj=np.array(m['uv'])[:,None];depth=np.ones((n,1))
    m['binding']=bind_memory(m,xyz,proj,depth,P)
    visibility=[dict(passed=True,share=.2) for _ in range(n)]
    appearance=[dict(ncc=1.,observed_std=10.,rendered_std=10.) for _ in range(n)]
    q=measurement(m,proj[:,0],depth[:,0],visibility,appearance,P)
    state=transition(None,q,P,True)
    return m,state,xyz,proj,depth

def test_canonical_binding_and_reciprocal():
    m,_,_,_,_=fixture();assert m['binding']['accepted'] and m['binding']['row']==0

def test_occlusion_suspends_without_losing_observations():
    m,s,xyz,proj,depth=fixture();before=deepcopy(m)
    q=measurement(m,proj[:,0],depth[:,0],[dict(passed=False) for _ in m['frames']],
        [dict(ncc=1.,observed_std=10.,rendered_std=10.) for _ in m['frames']],P)
    assert q['kind']=='unknown' and transition(s,q,P)['status']=='suspended'
    assert m==before

def test_revocation_and_recovery():
    _,s,_,_,_=fixture();bad=dict(s['measurement'],passed=False,kind='contradiction',confidence=0.)
    s=transition(s,bad,P);s=transition(s,bad,P);assert s['status']=='revoked'
    _,good,_,_,_=fixture();s=transition(s,good['measurement'],P);assert s['status']!='active'
    assert transition(s,good['measurement'],P)['status']=='active'

def test_keyframe_budget_fit_only_and_farthest():
    m,s,_,_,_=fixture();assert keyframes(0,[m],[s],P)==[14,12,10,8]
    assert keyframes(1,[m],[s],P)==[]

def test_disabled_identity_never_queries_model():
    base=torch.tensor(1.,requires_grad=True)
    out=add_memory(base,None,None,None,None,None,None,None,False)
    assert out[0] is base and out[1]==0

def test_joint_gradients_reach_current_and_distant_memory_frames():
    m,s,_,_,_=fixture();world=torch.tensor([[[100+i*.2+.1,80.,1.]] for i in range(16)],requires_grad=True)
    views=np.repeat(np.eye(4)[None],16,0)
    loss,aux,q=add_memory(world.sum()*0,lambda f:world[f],views,np.eye(3),0,[m],[s],P,True)
    assert q['endpoint_pairs']==4 and aux>0;loss.backward()
    assert world.grad.isfinite().all() and world.grad[0].abs().sum()>0 and world.grad[14].abs().sum()>0
    assert world.grad[1].abs().sum()==0

def test_suspended_has_no_auxiliary_gradient():
    m,s,_,_,_=fixture();s['status']='suspended';base=torch.tensor(1.,requires_grad=True)
    loss,aux,q=add_memory(base,None,None,None,0,[m],[s],P,True)
    assert aux==0 and q['endpoint_pairs']==0

def test_behind_camera_does_not_receive_fusion():
    m,s,_,_,_=fixture();world=torch.ones((16,1,3),requires_grad=True);world=world*torch.tensor([1.,1.,-1.])
    out=add_memory(world.sum()*0,lambda f:world[f],np.repeat(np.eye(4)[None],16,0),np.eye(3),0,[m],[s],P,True)
    assert out[1]==0 and out[2]['memories']==0
