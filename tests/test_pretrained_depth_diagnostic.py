import numpy as np
from experiments.staged_reconstruction.S1_pretrained_geometry.oracle_diagnostic.core import held_blocks, oracle_shape, ordinal_pairs, pair_counts


def fixture():
    y, x = np.mgrid[:40, :64]
    z = 3+x*.01+y*.005
    return z, ((1/z-.1)/.02).astype(np.float32), np.ones(z.shape, bool)


def test_oracle_recovers_affine_without_exporting_coefficients():
    z, raw, visible = fixture(); out, held, passed = oracle_shape(raw, z, visible)
    assert passed and abs(out[held]-z[held]).max() < 1e-6
    assert held.dtype == bool


def test_oracle_check_pixels_do_not_fit_coefficients():
    z, raw, visible = fixture(); held = held_blocks(*z.shape)
    out, _, _ = oracle_shape(raw, z, visible)
    changed = z.copy(); changed[held] += .5
    after, _, _ = oracle_shape(raw, changed, visible)
    assert np.array_equal(after, out)


def test_reversed_or_constant_prior_does_not_get_repaired():
    z, raw, visible = fixture()
    assert not oracle_shape(-raw, z, visible)[2]
    assert not oracle_shape(np.ones_like(raw), z, visible)[2]


def test_pair_generation_is_label_free_deduplicated_and_inside():
    pairs = ordinal_pairs(144, 256)
    assert (pairs[:, 0] < pairs[:, 1]).all() and pairs.max() < 144*256
    assert len(np.unique(pairs, axis=0)) == len(pairs)
    assert np.array_equal(pairs, ordinal_pairs(144, 256))


def test_exact_order_and_sign_reversal_are_detected():
    z, raw, visible = fixture(); zero = np.zeros(z.shape, bool)
    pairs = ordinal_pairs(*z.shape)
    good = pair_counts(raw, pairs, z, visible, visible, zero, zero)
    bad = pair_counts(-raw, pairs, z, visible, visible, zero, zero)
    assert good['static_ordered'] > 0 and good['static_ordered_wrong'] == 0
    assert bad['static_ordered_wrong'] == bad['static_ordered']


def test_empty_and_dynamic_endpoints_not_static():
    z, raw, visible = fixture(); empty = visible.copy(); visible[:] = False
    counts = pair_counts(raw, ordinal_pairs(*z.shape), z, visible, visible, empty, visible)
    assert counts['strong_pairs'] > 0 and counts['strong_static_visible'] == 0
    assert counts['strong_has_empty_endpoint'] == counts['strong_pairs']
