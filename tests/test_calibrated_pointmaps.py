"""Geometric recovery tests independent of any pretrained pointmap model."""
import numpy as np
import pytest

from variants.staged_reconstruction.calibrated_pointmaps.core import calibrate_scale, projection_errors


def fixture_points(seed=19, count=768, scale=3.7):
    rng = np.random.default_rng(seed)
    k = np.array([[570., 0, 255.5], [0, 570., 143.5], [0, 0, 1]])
    pixels = rng.uniform([30, 20], [480, 260], size=(count, 2))
    target = np.c_[pixels, np.ones(count)] @ np.linalg.inv(k).T * rng.uniform(4, 8, size=(count, 1))
    angle = .16
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    translation = np.array([.7, -.13, .08])
    source = (target - translation) @ rotation
    return source / scale, pixels, k, rotation, translation


def test_known_translation_and_rotation_recover_metric_pair_scale():
    raw, pixels, k, rotation, translation = fixture_points()
    result = calibrate_scale(raw, pixels, k, rotation, translation)
    assert result['scale_observable']
    assert result['scale'] == pytest.approx(3.7, rel=1e-10)
    assert np.max(projection_errors(raw, pixels, k, rotation, translation, result['scale'])) < 1e-9


def test_target_points_can_move_independently_of_first_time_shape():
    # Only the second-time pointmap/rays are involved: no cross-time point
    # equality or shared static-object correspondence enters the estimator.
    raw, pixels, k, rotation, translation = fixture_points(seed=231, scale=.61)
    result = calibrate_scale(raw, pixels, k, rotation, translation)
    assert result['scale'] == pytest.approx(.61, rel=1e-10)


def test_uniform_raw_pointmap_rescale_changes_fitted_gauge_only():
    raw, pixels, k, rotation, translation = fixture_points()
    a = calibrate_scale(raw, pixels, k, rotation, translation)
    b = calibrate_scale(raw * 9, pixels, k, rotation, translation)
    np.testing.assert_allclose(a['scale'] * raw, b['scale'] * raw * 9, atol=1e-10)


def test_robust_estimator_handles_wrong_pixels_without_optimizing_camera():
    raw, pixels, k, rotation, translation = fixture_points()
    corrupt = pixels.copy(); corrupt[::7] += [130, -90]
    original_rotation = rotation.copy(); original_translation = translation.copy()
    result = calibrate_scale(raw, corrupt, k, rotation, translation)
    assert result['scale'] == pytest.approx(3.7, rel=.01)
    np.testing.assert_array_equal(rotation, original_rotation)
    np.testing.assert_array_equal(translation, original_translation)
    held = np.arange(len(raw)) % 7 != 0
    assert np.median(projection_errors(raw[held], pixels[held], k, rotation, translation, result['scale'])) < .5


def test_pure_rotation_does_not_falsely_identify_scale():
    raw, pixels, k, rotation, _ = fixture_points()
    result = calibrate_scale(raw, pixels, k, rotation, np.zeros(3))
    assert not result['scale_observable'] and result['scale'] is None


def test_translation_parallel_to_all_rays_is_degenerate():
    k = np.eye(3); raw = np.tile([0., 0., 4.], (256, 1)); pixels = np.zeros((256, 2))
    result = calibrate_scale(raw, pixels, k, np.eye(3), np.array([0., 0., .5]))
    assert not result['scale_observable'] and result['scale'] is None


def test_known_intrinsic_mismatch_remains_a_measurable_error():
    raw, pixels, k, rotation, translation = fixture_points()
    wrong = k.copy(); wrong[0, 0] *= 1.6; wrong[1, 1] *= 1.6
    errors = projection_errors(raw, pixels, wrong, rotation, translation, 3.7)
    assert np.median(errors) > 20


def test_held_out_bad_geometry_cannot_be_hidden_by_fit_pixels():
    raw, pixels, k, rotation, translation = fixture_points()
    fit = np.arange(len(raw)) % 3 == 0
    result = calibrate_scale(raw[fit], pixels[fit], k, rotation, translation)
    bad = raw[~fit].copy(); bad[:, 0] += .5
    assert np.median(projection_errors(bad, pixels[~fit], k, rotation, translation, result['scale'])) > 50


def test_insufficient_and_nonfinite_input_is_not_accepted():
    raw, pixels, k, rotation, translation = fixture_points()
    assert not calibrate_scale(raw[:8], pixels[:8], k, rotation, translation)['scale_observable']
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        calibrate_scale(raw, pixels, k, rotation, translation)


def test_nonrigid_camera_and_invalid_intrinsics_are_rejected():
    raw, pixels, k, rotation, translation = fixture_points()
    with pytest.raises(ValueError, match='rotation'):
        calibrate_scale(raw, pixels, k, rotation * 2, translation)
    k[0, 0] = 0
    with pytest.raises(ValueError, match='intrinsic'):
        calibrate_scale(raw, pixels, k, rotation, translation)
