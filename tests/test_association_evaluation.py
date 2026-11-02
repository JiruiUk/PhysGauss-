import numpy as np
from variants.association_training.evaluate import candidate


def test_perfect_self_permutation_is_never_physical_identity_confirmation():
    m = np.eye(4) * 5
    c = candidate(m, np.eye(4))
    assert c['all_four_groups_supported']
    assert c['best_component_permutation'] == [0, 1, 2, 3]
    assert not c['physical_ID_confirmed']


def test_absent_groups_cannot_gain_support_from_arbitrary_assignment():
    c = candidate(np.zeros((4, 4)), np.eye(4))
    assert not c['all_four_groups_supported']
    assert not any(c['active_groups'])
    assert c['runner_up_gap_mass'] == 0


def test_four_groups_on_one_component_are_not_four_object_reidentification():
    m = np.zeros((4, 4)); m[:, 0] = 5
    c = candidate(m, np.ones((4, 4)))
    assert not c['all_four_groups_supported']


def test_ambiguous_contributions_preserve_zero_candidate_gap():
    c = candidate(np.ones((4, 4)), np.ones((4, 4)))
    assert c['runner_up_gap_mass'] == 0
    assert not c['all_four_groups_supported']


def test_footprint_without_projected_centers_does_not_pass_support_gate():
    c = candidate(np.eye(4) * 5, np.zeros((4, 4)))
    assert not c['all_four_groups_supported']
