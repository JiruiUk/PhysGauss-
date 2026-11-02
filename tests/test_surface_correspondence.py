import numpy as np
from variants.surface_correspondence.model import Config,particle_support,freeze_bank,assess_cohort,optional_bank


def test_flags_off_does_not_access_state():
    assert optional_bank() is None


def test_center_only_does_not_confirm_invisible_particle():
    assert not particle_support([0.],[0.],[True])[0]


def test_visible_footprint_does_not_override_off_fragment_center():
    assert not particle_support([1.],[1.],[False])[0]


def test_cross_fragment_footprint_stays_unknown():
    assert not particle_support([.5],[1.],[True])[0]


def test_flow_disagreement_rejects_good_instantaneous_overlap():
    assert not particle_support([1.],[1.],[True],[3.])[0]


def test_seed_ids_are_intersection_not_union_or_nearest():
    c=Config(enabled=True,min_particles=1)
    bank=freeze_bank([np.array([1,1,0],bool),np.array([0,1,1],bool),np.array([0,1,0],bool)],np.eye(3),c)
    assert bank['kernel_ids'].tolist()==[1]
    assert np.array_equal(bank['canonical'],[[0,1,0]])


def test_three_observations_required_and_weak_bank_rejected():
    import pytest
    with pytest.raises(ValueError):freeze_bank([np.ones(3,bool)],np.eye(3),Config(enabled=True))
    assert freeze_bank([np.ones(3,bool)]*3,np.eye(3),Config(enabled=True)) is None


def test_terminated_reference_cannot_be_reidentified_by_good_geometry():
    r=assess_cohort([1.],[1.],[True],[0.],False,Config(enabled=True))
    assert not r['accepted'] and 'image_fragment_unverified_or_terminated' in r['unknown_reasons']


def test_joint_support_gate_accepts_coherent_partial_surface():
    r=assess_cohort([1.],[1.],[True],[.5],True,Config(enabled=True))
    assert r['accepted']


def test_nonfinite_contribution_cannot_pass():
    assert not assess_cohort([np.nan],[1.],[True],[0.],True,Config(enabled=True))['accepted']


def test_fragment_loss_is_latched_and_not_nearest_reassigned():
    from variants.surface_correspondence.model import qualify_fragments
    components=np.zeros((10,10),int);labels=np.full((10,10),-1)
    mapping,alive=qualify_fragments(labels,components,[True]*4)
    assert (mapping==-1).all() and not alive.any()
    labels[:]=0;mapping,_=qualify_fragments(labels,components,alive)
    assert (mapping==-1).all()


def test_component_split_does_not_copy_one_seed_into_two_parts():
    from variants.surface_correspondence.model import qualify_fragments
    components=np.zeros((10,10),int);components[:,5:]=1;labels=np.zeros((10,10),int)
    mapping,alive=qualify_fragments(labels,components,[True]*4)
    assert mapping[0]==-1 and not alive[0]
