import json
from pathlib import Path
import numpy as np
from variants.staged_reconstruction.pretrained_depth.core import calibrate, affine_fit, split_anchors, metric_depth
from variants.staged_reconstruction.pretrained_depth.model import load_model, predict

P = json.loads((Path(__file__).resolve().parents[1] / 'experiments/staged_reconstruction/S1_pretrained_geometry/protocol.json').read_text())


def fixture():
    yy, xx = np.mgrid[:96, :128]
    depth = 3. + xx * .004 + yy * .006
    raw = ((1./depth - .1) / .015).astype(np.float32)
    sy, sx = np.mgrid[5:92:3, 5:124:3]
    uv = np.c_[sx.ravel(), sy.ravel()]
    return depth, raw, uv, depth[uv[:, 1], uv[:, 0]].copy()


def run(raw, uv, z):
    return calibrate(None, raw, uv, z, 1.98, 6.84, P, True)


def test_default_off_exact_identity_and_lazy_model():
    original = object()
    assert calibrate(original) == (original, None)
    assert predict(original) is original and load_model(None, None) is None


def test_known_affine_inverse_depth_recovered_dense():
    depth, raw, uv, z = fixture()
    out, info = run(raw, uv, z)
    assert info['calibrated'] and np.max(abs(out-depth)) < 1e-5
    assert abs(info['slope']-.015) < 1e-6 and abs(info['intercept']-.1) < 1e-6


def test_robust_fit_outliers_do_not_control_solution():
    _, raw, uv, z = fixture()
    fit = np.flatnonzero(~split_anchors(uv)); z[fit[::11]] += 2
    _, info = run(raw, uv, z)
    assert info['calibrated'] and info['held_MAE'] < 1e-4


def test_held_data_does_not_leak_into_fitting():
    _, raw, uv, z = fixture()
    _, before = run(raw, uv, z)
    z[split_anchors(uv)] += .3
    out, after = run(raw, uv, z)
    assert not out.any() and after['reason'] == 'held_anchor_disagreement'
    assert before['slope'] == after['slope'] and before['intercept'] == after['intercept']


def test_sparse_anchors_stay_unknown():
    _, raw, uv, z = fixture()
    out, info = run(raw, uv[:20], z[:20])
    assert not out.any() and info['reason'] == 'insufficient_anchors'


def test_planar_constant_depth_scale_is_degenerate():
    _, raw, uv, z = fixture(); z[:] = 4
    out, info = run(raw, uv, z)
    assert not out.any() and info['reason'] == 'degenerate_anchor_geometry'


def test_spatially_concentrated_fit_rejected():
    _, raw, uv, z = fixture(); uv[:, 0] = uv[:, 0] % 12
    out, info = run(raw, uv, z)
    assert not out.any() and info['reason'] == 'degenerate_anchor_geometry'


def test_reversed_rank_is_not_silently_flipped():
    _, raw, uv, z = fixture(); out, info = run(-raw, uv, z)
    assert not out.any() and info['reason'] == 'nonpositive_or_degenerate_affine_fit'


def test_constant_predictor_no_alternate_fit():
    _, raw, uv, z = fixture(); out, info = run(np.ones_like(raw), uv, z)
    assert not out.any() and info['reason'] == 'nonpositive_or_degenerate_affine_fit'


def test_invalid_inverse_depth_and_out_of_range_not_clamped():
    raw = np.array([[np.nan, np.inf, -1, 0, .1, 1/4, 1]], dtype=np.float32)
    depth, valid = metric_depth(raw, 1, 0, 2, 6)
    assert valid.tolist() == [[False, False, False, False, False, True, False]]
    assert depth[0, 5] == 4 and np.count_nonzero(depth) == 1


def test_anchor_split_and_subsample_are_deterministic_disjoint():
    _, raw, uv, z = fixture(); _, one = run(raw, uv, z); _, two = run(raw, uv, z)
    assert len(one['fit_ids']) == 256
    assert not np.intersect1d(one['fit_ids'], one['held_ids']).size
    assert np.array_equal(one['fit_ids'], two['fit_ids'])


def test_nonfinite_anchor_is_unknown_not_zero_constraint():
    _, raw, uv, z = fixture(); z[::20] = np.nan
    _, info = run(raw, uv, z)
    assert info['calibrated'] and info['held_MAE'] < 1e-4


def test_official_bgr_conversion_and_input_size():
    class Stub:
        def infer_image(self, image, input_size):
            assert image.tolist() == [[[30, 20, 10]]] and input_size == 518
            return np.array([[2.]], dtype=np.float32)
    assert predict(None, np.array([[[10, 20, 30]]], dtype=np.uint8), Stub(), enabled=True)[0, 0] == 2


def test_theilsen_positive_fit_and_pair_degeneracy():
    x = np.arange(10, dtype=float)
    assert affine_fit(x, 2*x+3) == (2., 3.)
    assert affine_fit(np.ones(10), x) is None
