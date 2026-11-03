"""Evaluator-only privileged diagnosis. Never imported by reconstruction."""
import numpy as np


def held_blocks(height, width):
    y, x = np.mgrid[:height, :width]
    return (x//8 + y//8) % 3 == 0


def oracle_shape(raw, gt_depth, visible):
    held = held_blocks(*raw.shape)
    fit = visible & ~held & np.isfinite(raw) & (gt_depth > 0)
    empty = np.zeros(raw.shape, dtype=np.float64)
    if fit.sum() < 2 or np.ptp(raw[fit]) <= 1e-8:
        return empty, held & visible, False
    matrix = np.c_[raw[fit].astype(np.float64), np.ones(fit.sum())]
    coefficients, *_ = np.linalg.lstsq(matrix, 1./gt_depth[fit].astype(np.float64), rcond=None)
    if not np.isfinite(coefficients).all() or coefficients[0] <= 0:
        return empty, held & visible, False
    inverse = coefficients[0] * raw.astype(np.float64) + coefficients[1]
    valid = np.isfinite(inverse) & (inverse > 0)
    np.divide(1., inverse, out=empty, where=valid)
    return empty, held & visible, True


def ordinal_pairs(height, width):
    y, x = np.mgrid[2:height:4, 2:width:4]
    x, y = x.ravel(), y.ravel()
    first = y * width + x
    pairs = []
    for dx, dy in [(4, 0), (0, 4), (16, 0), (0, 16), (64, 0), (0, 64)]:
        valid = (x+dx < width) & (y+dy < height)
        pairs.append(np.c_[first[valid], (y[valid]+dy)*width+x[valid]+dx])
    if len(first):
        pairs.append(np.c_[first, first[(np.arange(len(first))+997) % len(first)]])
    pairs = np.concatenate(pairs) if pairs else np.zeros((0, 2), int)
    pairs.sort(axis=1)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    return pairs


def pair_counts(raw, pairs, gt_depth, visible, thin, empty, dynamic):
    a, b = pairs.T
    flat = lambda v: v.ravel()
    gap = flat(raw)[a] - flat(raw)[b]
    span = float(np.diff(np.quantile(raw, [.05, .95]))[0])
    strong = abs(gap) >= .1*span if span > 1e-8 else np.zeros(len(pairs), bool)
    static = flat(visible)[a] & flat(visible)[b]
    z = flat(gt_depth)
    separation = abs(z[a]-z[b])
    ordered = static & (separation >= .05)
    thin_ordered = static & (flat(thin)[a] | flat(thin)[b]) & (separation >= .02)
    # Larger raw value predicts smaller depth. Exact ties fail separated-order tests.
    wrong = gap * (z[b]-z[a]) <= 0
    equal = static & (separation < .02)
    return {'pairs': len(pairs), 'strong_pairs': int(strong.sum()),
            'strong_static_visible': int((strong&static).sum()),
            'strong_has_empty_endpoint': int((strong&(flat(empty)[a] | flat(empty)[b])).sum()),
            'strong_has_dynamic_endpoint': int((strong&(flat(dynamic)[a] | flat(dynamic)[b])).sum()),
            'static_ordered': int(ordered.sum()), 'static_ordered_wrong': int((ordered&wrong).sum()),
            'strong_static_ordered': int((strong&ordered).sum()),
            'strong_static_ordered_wrong': int((strong&ordered&wrong).sum()),
            'thin_ordered': int(thin_ordered.sum()), 'thin_ordered_wrong': int((thin_ordered&wrong).sum()),
            'strong_thin_ordered': int((strong&thin_ordered).sum()),
            'strong_thin_ordered_wrong': int((strong&thin_ordered&wrong).sum()),
            'nearly_equal_static': int(equal.sum()), 'nearly_equal_static_strong': int((equal&strong).sum())}
