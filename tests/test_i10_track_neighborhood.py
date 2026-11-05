import numpy as np
import pytest
from variants.staged_reconstruction.i10_track_neighborhood.core import (
    reciprocal, patch_visibility, visibility_gate, neighborhood)


def config():
    return dict(equivalent_max_world_distance=.01, equivalent_max_projection_distance_px=.75,
                visibility_patch_radius=2, visibility_min_denominator=1e-8, visibility_min_share=.01,
                visibility_min_frames=8, visibility_min_held_frames=4,
                held_observations=[1, 3, 5, 7, 9, 11], neighbor_radius=.05, neighbor_limit=8,
                relative_displacement_rms_max=.02, minimum_neighbors=3)


def test_occluded_point_can_project_correctly_but_has_no_visible_support():
    alpha = np.ones((12, 12), np.float32)
    result = patch_visibility(alpha*0, alpha, [5, 5], config())
    assert result['known'] and not result['passed'] and result['share'] == 0


def test_vacuum_is_unknown_not_zero_error_or_background_label():
    result = patch_visibility(np.zeros((12, 12)), np.zeros((12, 12)), [5, 5], config())
    assert not result['known'] and result['share'] is None and not result['passed']


def test_fractional_patch_uses_full_alpha_denominator():
    alpha = np.full((12, 12), .6, np.float32)
    result = patch_visibility(alpha*.2, alpha, [5.25, 5.5], config())
    assert result['passed'] and abs(result['share']-.2) < 1e-6


def test_invalid_contribution_and_nan_rejected():
    a = np.ones((12, 12), np.float32)
    with pytest.raises(ValueError): patch_visibility(a*2, a, [5, 5], config())
    with pytest.raises(ValueError): patch_visibility(a*np.nan, a, [5, 5], config())
    with pytest.raises(ValueError): patch_visibility(-a, a, [5, 5], config())


def test_outside_patch_stays_unknown():
    a = np.ones((12, 12), np.float32)
    assert not patch_visibility(a, a, [0, 0], config())['known']


def test_held_visibility_required_even_when_eight_total_present():
    obs = [dict(passed=i not in [1, 3, 5, 7], known=True) for i in range(12)]
    assert not visibility_gate(obs, config())['passed']
    obs = [dict(passed=i < 8, known=True) for i in range(12)]
    assert visibility_gate(obs, config())['passed']


def original():
    return dict(fit_observations=list(range(0, 12, 2)), held_observations=list(range(1, 12, 2)),
                minimum_positive_depth=.01, binding_alternative_3D_distance=.05,
                binding_fit_rms_max_px=2., binding_held_rms_max_px=2., binding_all_max_px=4.,
                binding_fit_rms_alternative_margin_min_px=.25)


def test_independent_split_selects_distant_id_and_rejects_binding():
    xyz = np.zeros((12, 3, 3)); xyz[:, :, 2] = 4; xyz[:, 1, 0] = .1; xyz[:, 2, 0] = 1.
    predicted = np.zeros((12, 3, 2)); predicted[1::2, 0, 0] = 1.; predicted[::2, 1, 0] = 1.; predicted[:, 2, 0] = 5.
    track = dict(uv=np.zeros((12, 2)), binding=dict(row=0))
    result = reciprocal(track, xyz, predicted, xyz[..., 2], original(), config())
    assert not result['passed'] and result['swapped_binding']['row'] == 1


def test_nearby_same_surface_equivalent_is_explicitly_model_conditioned():
    xyz = np.zeros((12, 3, 3)); xyz[:, :, 2] = 4; xyz[:, 1, 0] = .005; xyz[:, 2, 0] = 1.
    predicted = np.zeros((12, 3, 2)); predicted[1::2, 0, 0] = .5; predicted[::2, 1, 0] = .5; predicted[:, 2, 0] = 5.
    result = reciprocal(dict(uv=np.zeros((12, 2)), binding=dict(row=0)), xyz, predicted, xyz[..., 2], original(), config())
    assert result['passed'] and not result['same_persistent_ID'] and result['surface_equivalence_is_model_conditioned']


def test_constant_translation_preserves_local_graph_and_excludes_self():
    xyz = np.zeros((12, 5, 3)); xyz[:, :, 0] = np.arange(5)*.01
    xyz += np.arange(12)[:, None, None]*np.array([.01, .02, -.01])
    result = neighborhood(xyz, 0, config())
    assert result['qualified'] and result['rows'] == [1, 2, 3, 4]


def test_temporary_contact_does_not_create_persistent_edge():
    xyz = np.zeros((12, 5, 3)); xyz[:, :, 0] = np.arange(5)*.01
    xyz[0, 4, 0] = .2
    result = neighborhood(xyz, 0, config())
    assert result['rows'] == [1, 2, 3] and 4 in result['candidate_rows']


def test_fewer_than_three_neighbors_remains_unqualified():
    xyz = np.zeros((12, 3, 3)); xyz[:, :, 0] = [0., .01, .02]
    assert not neighborhood(xyz, 0, config())['qualified']
