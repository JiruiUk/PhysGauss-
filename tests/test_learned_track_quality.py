import numpy as np
from variants.staged_reconstruction.learned_tracks.quality import qualify


def evidence():
    rng=np.random.default_rng(3)
    gray=rng.uniform(0,255,(20,20)).astype(np.float32)
    forward=np.broadcast_to(np.array([[[10.,10.]]]),(8,1,2)).copy()
    return forward,forward.copy(),np.ones((8,1),bool),np.ones((8,1),bool),[np.ones((20,20),np.float32)]*8,[gray]*8


CONFIG=dict(min_observations=8,patch_radius=2,closure_max_px=.75,direction_max_px=.75,ncc_min=.6,patch_std_min=2.)


def test_consistent_visible_track_passes():
    accepted,_=qualify(*evidence(),CONFIG)
    assert accepted.tolist()==[True]


def test_forced_reverse_query_does_not_fake_closure():
    values=list(evidence()); values[1][0,0,0]+=2
    accepted,info=qualify(*values,CONFIG)
    assert not accepted[0] and info['closure_px']==[2.]


def test_confidence_does_not_override_direction_disagreement():
    values=list(evidence()); values[1][3,0,0]+=2
    accepted,info=qualify(*values,CONFIG)
    assert not accepted[0] and not info['checks']['direction_agreement'][0]


def test_nonquery_invisibility_is_rejected():
    values=list(evidence()); values[2][4,0]=False
    assert not qualify(*values,CONFIG)[0][0]


def test_flat_texture_is_not_material_evidence():
    values=list(evidence()); values[-1]=[np.zeros((20,20),np.float32)]*8
    assert not qualify(*values,CONFIG)[0][0]
