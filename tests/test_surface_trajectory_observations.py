"""Observation contracts precede real data preparation and any training."""
import numpy as np
from variants.staged_reconstruction.surface_trajectory.observations import sample, step, balanced_subset, extract_window, appearance_masks


CONFIG = dict(patch_radius=2, fb_max_px=.75, ncc_min=.6, patch_std_min=2.)


def translated():
    rng = np.random.default_rng(7)
    first = rng.uniform(0, 255, (24, 32)).astype(np.float32)
    second = np.roll(first, 1, axis=1)
    forward = np.zeros((24, 32, 2), np.float32)
    forward[..., 0] = 1
    return first, second, forward, -forward, np.ones((24, 32), np.float32)


def test_pixel_sampling_uses_xy_not_row_column():
    field = np.arange(20, dtype=np.float32).reshape(4, 5)
    assert sample(field, np.array([[3., 2.]], np.float32))[0] == 13


def test_visible_translation_has_correct_direction_and_independent_patch_support():
    a, b, f, back, mask = translated()
    p = np.array([[12., 10.]], np.float32)
    q, good, closure, ncc = step(p, f, back, mask, mask, a, b, CONFIG)
    assert good.all() and np.allclose(q, p+[1, 0]) and closure[0] == 0 and ncc[0] > .99


def test_backward_disagreement_occlusion_and_out_of_bounds_are_rejected():
    a, b, f, back, mask = translated()
    points = np.array([[12., 10.], [31., 10.]], np.float32)
    assert not step(points, f, np.zeros_like(back), mask, mask, a, b, CONFIG)[1].any()
    assert not step(points, f, back, mask, mask*0, a, b, CONFIG)[1].any()


def test_consistent_flow_is_insufficient_for_featureless_or_wrong_appearance():
    a, b, f, back, mask = translated()
    points = np.array([[12., 10.]], np.float32)
    assert not step(points, f, back, mask, mask, a*0, b*0, CONFIG)[1].any()
    assert not step(points, f, back, mask, mask, a, 255-b, CONFIG)[1].any()


def test_balanced_sampling_does_not_infer_object_counts_and_is_reproducible():
    tracks = [{'time_band': b, 'source_frame': j} for b in range(3) for j in range(12)]
    first = balanced_subset([dict(t) for t in tracks], 21, 17)
    second = balanced_subset([dict(t) for t in tracks], 21, 17)
    assert first == second and [sum(t['time_band'] == b for t in first) for b in range(3)] == [7]*3
    assert not any('object' in key for t in first for key in t)


def test_shorter_valid_prefix_is_preserved_without_relaxing_eight_observation_or_closure_gate():
    rng = np.random.default_rng(19)
    first = rng.uniform(0, 255, (48, 72)).astype(np.float32)
    grays = [np.roll(first, j, axis=1) for j in range(16)]
    masks = [np.ones_like(first) for _ in grays]
    forward = np.zeros((48, 72, 2), np.float32)
    forward[..., 0] = 1
    backward = -forward*.9
    flows = [(forward, backward) for _ in range(15)]
    config = dict(CONFIG, min_observations=8, max_observations=16, max_corners=12,
        corner_quality=.01, seed_distance=4., fragment_closure_max_px=.75)
    original, _ = extract_window(0, list(range(16)), list(np.linspace(0, 1, 16)), grays, masks, flows, config)
    corrected, _ = extract_window(0, list(range(16)), list(np.linspace(0, 1, 16)), grays, masks, flows,
                                   dict(config, check_valid_prefixes=True))
    assert len(corrected) > len(original)
    assert all(len(t['frames']) >= 8 and t['reverse_closure_px'] <= .75 for t in corrected)


def test_highlight_neighborhood_is_excluded_while_small_dark_eye_remains_in_body_envelope():
    rgba = np.zeros((32, 48, 4), np.float32)
    rgba[..., 3] = 1.
    rgba[4:28, 4:44, :3] = [1., .8, .1]
    rgba[10:12, 12:14, :3] = [.05, .05, .05]
    rgba[16:18, 32:34, :3] = [1., 1., 1.]
    body, valid = appearance_masks(rgba, 2, True)
    assert body[10, 12] and valid[10, 12]
    assert body[16, 32] and not valid[16, 32] and not valid[15, 31]
