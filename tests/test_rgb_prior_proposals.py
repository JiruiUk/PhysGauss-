import json
from pathlib import Path
import numpy as np
import torch
from variants.staged_reconstruction.rgb_prior_proposals.core import ordinal_bounds, eligible_old_depth, accept_rgb, propose
from variants.staged_reconstruction.adaptive_static.geometry import patch_features

P = json.loads((Path(__file__).resolve().parents[1]/'experiments/staged_reconstruction/S1_rgb_prior_proposals/protocol.json').read_text())


def bounds_fixture():
    raw = np.tile(np.linspace(0, 1, 64), (64, 1))
    uv = np.array([[2, 32], [61, 32]])
    anchors = np.array([[24, 4], [28, 16], [32, 28], [36, 40], [40, 52]])
    lo = np.array([3.7, 3.8, 3.9, 4., 4.1]); hi = lo+.04
    return uv, raw, anchors, lo, hi


def test_bounds_sign_and_fixed_quantiles():
    uv, raw, a, lo, hi = bounds_fixture()
    lower, upper, known, lb, hb = ordinal_bounds(uv, raw, a, lo, hi, 2, 6, P)
    assert known.all() and lower[0] == np.quantile(lo, .8) and upper[1] == np.quantile(hi, .2)
    assert lower[1] == 2 and upper[0] == 6 and lb[0] == 5 and hb[1] == 5


def test_same_anchor_block_does_not_inflate_evidence():
    uv, raw, a, lo, hi = bounds_fixture(); a[:] = [28, 28]
    _, _, known, lb, hb = ordinal_bounds(uv, raw, a, lo, hi, 2, 6, P)
    assert not known.any() and lb.max() <= 1 and hb.max() <= 1


def test_missing_and_constant_prediction_are_unknown():
    uv, raw, a, lo, hi = bounds_fixture()
    assert not ordinal_bounds(uv, raw, a[:0], lo[:0], hi[:0], 2, 6, P)[2].any()
    assert not ordinal_bounds(uv, np.ones_like(raw), a, lo, hi, 2, 6, P)[2].any()


def test_contradictory_bounds_no_proposal():
    uv, raw, a, lo, hi = bounds_fixture(); uv[:] = [32, 32]
    a = np.array([[8, y] for y in [4, 16, 28, 40]]+[[56, y] for y in [4, 16, 28, 40]])
    lo = np.array([2.]*4+[5.]*4); hi = lo+.04
    lower, upper, known, *_ = ordinal_bounds(uv, raw, a, lo, hi, 2, 6, P)
    assert (lower > upper).all() and not known.any()


def test_existing_interval_inside_is_not_nominated():
    assert eligible_old_depth(np.array([4., 3., 5.]), np.array([3.5]*3), np.array([4.5]*3), np.array([True]*3)).tolist() == [False, True, True]


def test_anchor_selection_deterministic_spatial_and_capped():
    y, x = np.mgrid[4:60:2, 16:48:2]; a = np.c_[x.ravel(), y.ravel()]
    raw = np.tile(np.linspace(0, 1, 64), (64, 1)); uv = np.array([[2, 32]])
    lo = np.linspace(3.8, 4.1, len(a)); hi = lo+.04
    one = ordinal_bounds(uv, raw, a, lo, hi, 2, 6, P)
    ids = np.lexsort((a[:, 0], a[:, 1])); ids = ids[np.linspace(0, len(a)-1, 256).astype(int)]
    two = ordinal_bounds(uv, raw, a[ids], lo[ids], hi[ids], 2, 6, P)
    assert all(np.array_equal(v, w) for v, w in zip(one, two))


def accept_fixture(held_new=None, fit_new=None):
    old, new = torch.tensor([4.8]), torch.tensor([4.])
    fit_old = torch.tensor([.2]); fit_new = torch.tensor([.1]) if fit_new is None else fit_new
    held_old = torch.tensor([[.2], [.3]])
    held_new = torch.tensor([[.1], [.3]]) if held_new is None else held_new
    return old, new, fit_old, fit_new, held_old, held_new, torch.tensor([True])


def test_default_off_exact_identity_no_features_required():
    old = torch.tensor([4.])
    result, d = propose(None, old)
    assert result is old and not d['accepted'].any()
    args = accept_fixture(); same, mask = accept_rgb(*args)
    assert same is args[0] and not mask.any()


def test_held_one_strict_improvement_and_other_nonworse_accepts():
    out, mask = accept_rgb(*accept_fixture(), enabled=True)
    assert mask.all() and out[0] == 4


def test_held_one_worse_rejects_despite_better_fit():
    out, mask = accept_rgb(*accept_fixture(torch.tensor([[.1], [.31]])), enabled=True)
    assert not mask.any() and out[0] == torch.tensor(4.8)


def test_held_tie_does_not_admit_new_depth():
    _, mask = accept_rgb(*accept_fixture(torch.tensor([[.2], [.3]])), enabled=True)
    assert not mask.any()


def test_fit_worse_or_nonfinite_held_is_rejected():
    assert not accept_rgb(*accept_fixture(fit_new=torch.tensor([.21])), enabled=True)[1].any()
    assert not accept_rgb(*accept_fixture(torch.tensor([[float('nan')], [.1]])), enabled=True)[1].any()


def test_native_calibrated_textured_plane_proposal():
    height = width = 64
    # Integer source-pixel shifts at the true plane remove interpolation bias
    # in the analytic fixture; production cameras/preprocessing stay unchanged.
    views = torch.eye(4).repeat(7, 1, 1)
    views[:, 0, 3] = torch.tensor([0., 8/35, 16/35, 24/35, -8/35, -16/35, -24/35])
    k = torch.tensor([[35., 0, 31.5], [0, 35., 31.5], [0, 0, 1.]])
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    images = []
    for v in views:
        wx = (xx-31.5)/35*4-v[0, 3]; wy = (yy-31.5)/35*4
        g = .45+.15*torch.sin(wx*7)+.1*torch.cos(wy*9+wx*3)
        images.append(torch.stack([g, g*.9, g*.8]))
    features, texture = patch_features(torch.stack(images), 5)
    uv = torch.tensor([[32., 30.], [29., 35.]])
    old = torch.tensor([4.8, 4.8]); lo = torch.tensor([3.8, 3.8]); hi = torch.tensor([4.2, 4.2])
    result, diagnostics = propose(uv, old, lo, hi, torch.ones(2, dtype=torch.bool), features, texture, views, k, 0, list(range(1, 7)), 2, 6, True)
    assert diagnostics['accepted'].all() and (result-4).abs().max() < .03


def test_no_grid_sample_or_no_nomination_keeps_original(monkeypatch):
    import variants.staged_reconstruction.rgb_prior_proposals.core as core
    def bad(*args): raise AssertionError('No RGB access for no eligible points')
    monkeypatch.setattr(core, 'source_cost', bad)
    old = torch.tensor([4.]); uv = torch.tensor([[2., 2.]])
    result, d = propose(uv, old, torch.tensor([3.]), torch.tensor([5.]), torch.tensor([True]), sources=list(range(6)), enabled=True)
    assert torch.equal(result, old) and not d['accepted'].any()


def test_nonempty_interval_without_grid_hypothesis_retains_old(monkeypatch):
    import variants.staged_reconstruction.rgb_prior_proposals.core as core
    monkeypatch.setattr(core, 'source_cost', lambda points, *args: (points[..., 2]-4).abs())
    old = torch.tensor([4.8]); uv = torch.tensor([[2., 2.]])
    features = torch.ones(7, 28, 5, 5); texture = torch.ones(7, 5, 5, dtype=torch.bool)
    out, d = propose(uv, old, torch.tensor([4.00001]), torch.tensor([4.00002]), torch.tensor([True]),
                     features, texture, torch.eye(4).repeat(7, 1, 1), torch.eye(3), 0, list(range(1, 7)), 2, 6, True)
    assert torch.equal(out, old) and not d['accepted'].any() and torch.isnan(d['proposed']).all()
