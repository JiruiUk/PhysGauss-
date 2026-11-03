"""Train-anchor affine calibration of relative inverse depth, no GT access."""
import numpy as np


def split_anchors(uv):
    return ((uv[:, 0] // 8 + uv[:, 1] // 8) % 3) == 0


def affine_fit(x, y):
    """Deterministic Theil-Sen; no optimization or alternate estimator."""
    a, b = np.triu_indices(len(x), 1)
    dx = x[b] - x[a]
    good = np.abs(dx) > 1e-6 * max(1., float(np.ptp(x)))
    if not good.any():
        return None
    slope = float(np.median((y[b[good]] - y[a[good]]) / dx[good]))
    intercept = float(np.median(y - slope * x))
    if not np.isfinite([slope, intercept]).all() or slope <= 0:
        return None
    return slope, intercept


def metric_depth(raw, slope, intercept, near, far):
    inv = slope * raw.astype(np.float64) + intercept
    valid = np.isfinite(inv) & (inv > 0)
    z = np.zeros_like(inv)
    np.divide(1., inv, out=z, where=valid)
    valid &= (z >= near) & (z <= far)
    return np.where(valid, z, 0).astype(np.float32), valid


def calibrate(original, raw=None, uv=None, anchor_depth=None, near=None, far=None,
              protocol=None, enabled=False):
    if not enabled:
        return original, None
    p = protocol
    assert raw.ndim == 2 and np.asarray(uv).shape == (len(anchor_depth), 2)
    height, width = raw.shape
    uv = np.asarray(uv, dtype=np.int64)
    assert ((uv >= 0).all() and (uv[:, 0] < width).all() and (uv[:, 1] < height).all())
    anchor_depth = np.asarray(anchor_depth, dtype=np.float64)
    x = raw[uv[:, 1], uv[:, 0]].astype(np.float64)
    finite = np.isfinite(x) & np.isfinite(anchor_depth) & (anchor_depth > 0)
    held = split_anchors(uv)
    fit_ids = np.flatnonzero(finite & ~held)
    fit_ids = fit_ids[np.lexsort((uv[fit_ids, 0], uv[fit_ids, 1]))]
    if len(fit_ids) > p['maximum_fit_anchors']:
        fit_ids = fit_ids[np.linspace(0, len(fit_ids)-1, p['maximum_fit_anchors']).astype(int)]
    held_ids = np.flatnonzero(finite & held)
    result = {'calibrated': False, 'reason': 'insufficient_anchors',
              'anchors': len(uv), 'fit_count': len(fit_ids), 'held_count': len(held_ids),
              'fit_ids': fit_ids, 'held_ids': held_ids}
    empty = np.zeros(raw.shape, dtype=np.float32)
    if len(fit_ids) < p['minimum_fit_anchors'] or len(held_ids) < p['minimum_held_anchors']:
        return empty, result
    extent = np.ptp(uv[fit_ids], axis=0) / np.array([width, height])
    inverse = 1. / anchor_depth[fit_ids]
    spread = float(np.diff(np.quantile(inverse, [.1, .9]))[0])
    result.update({'fit_image_extent_xy': extent.tolist(), 'fit_inverse_depth_spread': spread})
    if (extent < p['minimum_anchor_image_extent_fraction_xy']).any() or spread < p['minimum_fit_inverse_depth_q90_q10']:
        result['reason'] = 'degenerate_anchor_geometry'
        return empty, result
    fit = affine_fit(x[fit_ids], inverse)
    if fit is None:
        result['reason'] = 'nonpositive_or_degenerate_affine_fit'
        return empty, result
    slope, intercept = fit
    result.update({'slope': slope, 'intercept': intercept})
    inv_held = slope * x[held_ids] + intercept
    if not np.isfinite(inv_held).all() or (inv_held <= 0).any():
        result['reason'] = 'nonpositive_held_inverse_depth'
        return empty, result
    error = np.abs(1. / inv_held - anchor_depth[held_ids])
    result.update({'held_MAE': float(error.mean()), 'held_P90': float(np.quantile(error, .9))})
    if result['held_MAE'] > p['maximum_held_depth_MAE'] or result['held_P90'] > p['maximum_held_depth_P90']:
        result['reason'] = 'held_anchor_disagreement'
        return empty, result
    depth, _ = metric_depth(raw, slope, intercept, near, far)
    result.update({'calibrated': True, 'reason': 'passed_train_anchor_check',
                   'valid_pixels': int((depth > 0).sum())})
    return depth, result
