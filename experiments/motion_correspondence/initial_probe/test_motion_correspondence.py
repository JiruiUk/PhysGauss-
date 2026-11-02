import numpy as np
import torch
from variants.motion_correspondence.model import Config,State,Observation,MotionTracker,predict,is_merged,optional_visible_propagation


def obs(x,y=20,component=0):return Observation(np.array([float(x),float(y)]),25.,component)


def test_time_normalization_and_irregular_prediction():
    c=Config(acceleration_sigma=0);s=State(np.array([10.,20.,199.,0.]),np.eye(4),25.,0.)
    p=predict(s,3/199,c);assert np.allclose(p.mean[:2],[13,20])


def test_missing_observation_increases_uncertainty():
    tr=MotionTracker([obs(10)],0.)
    tr.step([],2/199);p=tr.hypotheses[0]['states'][0].covariance
    tr.step([],5/199);q=tr.hypotheses[0]['states'][0].covariance
    assert np.trace(q[:2,:2])>np.trace(p[:2,:2]) and tr.last_result['accepted_assignment']==[-1]


def test_history_predicts_passing_not_nearest_last_position():
    tr=MotionTracker([obs(10),obs(50,component=1)],0.,Config(uncertainty_guard=False))
    for n in range(1,5):tr.step([obs(10+2*n),obs(50-2*n,component=1)],n/199)
    tr.step([],10/199)
    r=tr.step([obs(34),obs(26,component=1)],12/199)
    assert r['best_assignment']==[0,1]


def test_symmetric_return_keeps_ambiguity():
    tr=MotionTracker([obs(15),obs(25,component=1)],0.,Config(initial_velocity_sigma=0,acceleration_sigma=0,uncertainty_guard=True))
    tr.step([],1/199)
    # Symmetric candidates at equal likelihood for both stationary means.
    r=tr.step([obs(20,15),obs(20,25,component=1)],4/199)
    assert r['accepted_assignment']==[-1,-1]
    assert r['retained_hypotheses']>1


def test_merged_observation_does_not_get_single_ID():
    mask=np.zeros((50,50),bool);mask[15:25,10:30]=True
    merged=Observation(np.array([20.,20.]),200.,0,mask)
    tr=MotionTracker([obs(15),obs(25,component=1)],0.)
    assert is_merged(merged,tr.hypotheses[0]['states'],tr.config)
    r=tr.step([merged],1/199);assert r['accepted_assignment']==[-1,-1]


def test_one_observation_cannot_get_two_IDs():
    tr=MotionTracker([obs(15),obs(25,component=1)],0.)
    r=tr.step([obs(20)],2/199)
    a=[j for j in r['best_assignment'] if j>=0];assert len(a)==len(set(a))


def test_far_unknown_is_not_forced_to_nearest_track():
    tr=MotionTracker([obs(10)],0.,Config(initial_velocity_sigma=0,acceleration_sigma=0))
    r=tr.step([obs(350)],1/199);assert r['accepted_assignment']==[-1]


def test_expired_track_not_relabelled():
    tr=MotionTracker([obs(10)],0.,Config(max_missing_frames=5))
    r=tr.step([obs(10)],6/199);assert r['accepted_assignment']==[-1]


def test_flags_off_exactly_delegates():
    from output.surface_support.core import propagate
    a=np.full((8,8),-1);a[2:5,2:5]=0;c=np.zeros((8,8),dtype=int);f=torch.zeros(8,8,2)
    expected=propagate(a,c,f,f);actual=optional_visible_propagation(a,c,f,f,enabled=False)
    assert np.array_equal(expected[0],actual[0]) and expected[1]==actual[1]


def test_resume_keeps_all_hypotheses_and_scores():
    import json
    tr=MotionTracker([obs(10),obs(40,component=1)],0.)
    tr.step([obs(12),obs(38,component=1)],1/199);tr.step([],2/199)
    restored=MotionTracker.from_state_dict(json.loads(json.dumps(tr.state_dict())))
    actual=tr.step([obs(16),obs(34,component=1)],3/199)
    assert restored.step([obs(16),obs(34,component=1)],3/199)==actual


def test_large_single_duck_is_not_marked_merged_by_other_ducks_size():
    mask=np.zeros((50,50),bool);mask[10:30,5:25]=True
    large=Observation(np.array([15.,20.]),400.,0,mask)
    tr=MotionTracker([Observation(np.array([15.,20.]),400.,0),obs(40,component=1)],0.)
    assert not is_merged(large,tr.hypotheses[0]['states'],tr.config)
