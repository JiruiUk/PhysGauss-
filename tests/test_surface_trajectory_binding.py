import numpy as np
import pytest
from variants.staged_reconstruction.surface_trajectory.binding import fit_weights, depth_clusters, choose_support


def test_weighted_3d_projection_is_not_average_pixel_projection():
    # Pixel x of this 3D midpoint is 2/3, whereas mean projected x is 1/2.
    h = np.array([[[0., 0., 0., 1.], [2., 0., 0., 2.]]]*3)
    target = np.array([[(2/3+1)*200-.5, 112.]]*3)
    weights, error = fit_weights(h, target)
    assert np.allclose(weights, [.5, .5], atol=1e-5) and error.max() < 1e-4


def test_complete_depth_range_does_not_chain_connect_two_separate_surfaces():
    depths = np.array([[1., 1., 1.], [1.015, 1.015, 1.015], [1.03, 1.03, 1.03]])
    groups = depth_clusters(depths, .02)
    assert groups == [[0, 1], [2]]


def record(depth, h):
    return dict(valid=True, distance=0., total_mass=1., foreground_mass=1., patch_mass=.2,
                depth=depth, homogeneous=h)


CONFIG = dict(candidate_radius_px=3., minimum_total_mass=.02, foreground_purity=.95,
              minimum_patch_mass=.005, depth_tolerance=.02, maximum_support_rows=8,
              bootstrap_max_error_px=1., ambiguity_gap_px=.25)


def test_two_comparable_depth_hypotheses_remain_unknown():
    observations = [{1:record(1., [0., 0., 0., 1.]), 2:record(2., [0., 0., 0., 2.])} for _ in range(3)]
    best, rejection = choose_support(observations, [[199.5, 112.]]*3, CONFIG)
    assert best is None and 'Multiple depth' in rejection['reason']


def test_future_observations_cannot_be_used_for_binding_admission():
    with pytest.raises(ValueError, match='three bootstrap'):
        choose_support([{}]*8, [[0, 0]]*8, CONFIG)


def test_invisible_row_is_not_assigned_despite_perfect_center_projection():
    observations = [{1:dict(record(1., [0., 0., 0., 1.]), patch_mass=0.)} for _ in range(3)]
    best, rejection = choose_support(observations, [[199.5, 112.]]*3, CONFIG)
    assert best is None and 'No row' in rejection['reason']
