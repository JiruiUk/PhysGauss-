"""A metric pair-scale estimate with known cameras, no static-point assumption."""
import numpy as np


def error_quantiles(errors):
    # Preserve behind-camera failures without producing NaN via inf-inf
    # interpolation inside np.quantile. Null is an unavailable/failing metric.
    values = np.sort(errors)
    out = []
    for q in [.5, .9]:
        index = q * (len(values)-1)
        lo, hi = int(np.floor(index)), int(np.ceil(index))
        value = values[lo] if lo == hi else (1-index+lo)*values[lo] + (index-lo)*values[hi]
        out.append(float(value) if np.isfinite(value) else None)
    return out


def validate(points, pixels, k, rotation, translation):
    points, pixels, k, rotation, translation = [np.asarray(v, dtype=np.float64) for v in (points, pixels, k, rotation, translation)]
    if points.ndim != 2 or points.shape[1] != 3 or pixels.shape != (len(points), 2):
        raise ValueError('Expected corresponding Nx3 points and Nx2 pixel rays')
    if not all(np.isfinite(v).all() for v in (points, pixels, k, rotation, translation)):
        raise ValueError('Inputs must be finite')
    if k.shape != (3, 3) or k[0, 0] <= 0 or k[1, 1] <= 0 or abs(np.linalg.det(k)) < 1e-10:
        raise ValueError('Invalid intrinsic matrix')
    if rotation.shape != (3, 3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
        raise ValueError('Expected a rigid camera rotation')
    if translation.shape != (3,):
        raise ValueError('Expected three-vector camera translation')
    return points, pixels, k, rotation, translation


def projection_errors(points, pixels, k, rotation, translation, scale):
    points, pixels, k, rotation, translation = validate(points, pixels, k, rotation, translation)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be finite and positive')
    target = (points * scale) @ rotation.T + translation
    homogeneous = target @ k.T
    valid = target[:, 2] > 1e-8
    error = np.full(len(points), np.inf)
    error[valid] = np.linalg.norm(homogeneous[valid, :2] / homogeneous[valid, 2:3] - pixels[valid], axis=1)
    return error


def calibrate_scale(points, pixels, k, rotation, translation, *, minimum_support=128, cauchy_relative_baseline=.05, iterations=8):
    """Solve b x (R (s Q_j) + t)=0 for one positive pair scale.

    Q_j is the model's second-time pointmap in the first camera coordinate
    system, indexed by second-image pixels. Thus no equality between moving
    points at different times is assumed. Cameras are never optimization
    variables. A positive estimate is not a certificate of accurate geometry.
    """
    points, pixels, k, rotation, translation = validate(points, pixels, k, rotation, translation)
    if minimum_support < 2 or iterations < 1 or cauchy_relative_baseline <= 0:
        raise ValueError('Invalid robust estimator settings')
    baseline = float(np.linalg.norm(translation))
    failed = {'scale_observable': False, 'scale': None, 'camera_baseline': baseline, 'fit_support': len(points)}
    if len(points) < minimum_support:
        return {**failed, 'reason': 'Insufficient fit support'}
    if baseline < 1e-4:
        return {**failed, 'reason': 'Camera translation too small to identify pair scale'}
    rays = np.c_[pixels, np.ones(len(points))] @ np.linalg.inv(k).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    rotated = points @ rotation.T
    a = np.cross(rays, rotated); d = np.cross(rays, translation)
    aa = (a*a).sum(1); ad = (a*d).sum(1)
    valid = aa > 1e-14
    if valid.sum() < minimum_support:
        return {**failed, 'reason': 'Degenerate ray/baseline geometry'}
    # Median initialization protects the scalar solve against isolated
    # high-leverage prediction errors, without RANSAC or RNG use.
    scale = float(np.median(-ad[valid] / aa[valid]))
    if scale <= 0 or not np.isfinite(scale):
        return {**failed, 'reason': 'No positive robust scale initialization'}
    for _ in range(iterations):
        residual = np.linalg.norm(scale*a+d, axis=1) / baseline
        weights = 1 / (1 + (residual/cauchy_relative_baseline)**2)
        denominator = float(np.sum(weights[valid] * aa[valid]))
        if denominator <= 1e-14:
            return {**failed, 'reason': 'Numerically unobservable weighted scale'}
        scale = float(-np.sum(weights[valid] * ad[valid]) / denominator)
        if scale <= 0 or not np.isfinite(scale):
            return {**failed, 'reason': 'Robust solve has nonpositive scale'}
    direction = rotated / np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
    angles = np.degrees(np.arccos(np.clip((direction*rays).sum(1), -1, 1)))
    return {'scale_observable': True, 'scale': scale, 'camera_baseline': baseline,
            'fit_support': len(points), 'median_predicted_ray_parallax_degrees': float(np.median(angles)),
            'normalized_cross_residual_median_p90': np.quantile(np.linalg.norm(scale*a+d, axis=1)/baseline, [.5, .9]).tolist(),
            'fit_projection_error_median_p90_pixels': error_quantiles(projection_errors(points, pixels, k, rotation, translation, scale))}
