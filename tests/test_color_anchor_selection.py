"""Spatial and temporal anchor rules cannot fill missing observations."""
import numpy as np
import pytest
from variants.staged_reconstruction.uncertainty_motion.selection import select_seed, compare_seeds


def fixture():
    support = np.tile(np.array([9., 8., 7., 6., .24, 10.]), (6, 1))
    good = np.ones_like(support); good[:, -1] = .9
    xyz = np.tile(np.array([[0., 0., 0.], [.01, 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.], [4., 0., 0.]]), (6, 1, 1))
    return support, good, np.ones(6), xyz


def test_spatial_sampling_keeps_best_supported_per_voxel_and_recurring_only():
    support, good, opacity, xyz = fixture()
    support[4:, 3] = 0
    r = select_seed(support, good, opacity, xyz, fraction=1., voxel_size=.1)
    assert r['anchors'].tolist() == [True, False, True, False, False, False]
    assert r['period'].tolist() == [6, 6, 6, 4, 0, 0]


def test_threshold_qualification_is_independent_for_each_seed():
    support, good, opacity, xyz = fixture()
    a = select_seed(support, good, opacity, xyz, fraction=1., voxel_size=.1)
    support[:, 4] = .26
    b = select_seed(support, good, opacity, xyz, fraction=1., voxel_size=.1)
    assert not a['per_frame'][:, 4].any() and b['per_frame'][:, 4].all()
    assert np.all(compare_seeds(a['per_frame'], b['per_frame']) < 1)


def test_empty_selections_do_not_pass_agreement():
    empty = np.zeros((6, 20), dtype=bool)
    assert np.array_equal(compare_seeds(empty, empty), np.zeros(6))


def test_recurrence_below_five_is_rejected_without_filling_anchor_budget():
    support, good, opacity, xyz = fixture(); good[4:] = 0
    r = select_seed(support, good, opacity, xyz, fraction=1., voxel_size=.1)
    assert not r['anchors'].any()


@pytest.mark.parametrize('size', [0., -.1, float('nan')])
def test_invalid_spatial_scale_rejected(size):
    with pytest.raises(ValueError):
        select_seed(*fixture(), fraction=1., voxel_size=size)
