"""Camera-compensated stationarity evidence, independent of color classes."""
import numpy as np


def triangulate(uv, matrices):
    rows = []
    for point, projection in zip(uv, matrices):
        rows.extend([point[0]*projection[2]-projection[0], point[1]*projection[2]-projection[1]])
    _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
    if abs(vh[-1, 3]) < 1e-12:
        return np.full(3, np.nan)
    return vh[-1, :3]/vh[-1, 3]


def evidence(ids, uv, views, intrinsic, min_observations=12, min_span=30,
             min_angle=2., median_limit=1.5, p90_limit=3.):
    """Fit alternating observations, test the others; keep uncertain tracks unknown."""
    ids = np.asarray(ids)
    uv = np.asarray(uv, dtype=float)
    result = {'label': 'unknown', 'observations': len(ids),
              'frame_span': int(ids[-1]-ids[0]) if len(ids) else 0}
    if len(ids) < min_observations or result['frame_span'] < min_span:
        result['reason'] = 'insufficient temporal evidence'
        return result
    matrices = np.asarray([intrinsic@views[int(n)][:3] for n in ids])
    xyz = triangulate(uv[::2], matrices[::2])
    if not np.isfinite(xyz).all():
        return {**result, 'reason': 'degenerate triangulation'}
    cameras = np.asarray([np.linalg.inv(views[int(n)])[:3, 3] for n in ids])
    rays = xyz-cameras
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    angle = float(np.degrees(np.arccos(np.clip((rays@rays.T).min(), -1, 1))))
    projection = np.einsum('nij,j->ni', matrices, np.r_[xyz, 1.])
    depths = np.asarray([(views[int(n)]@np.r_[xyz, 1.])[2] for n in ids])
    errors = np.linalg.norm(projection[:, :2]/projection[:, 2:]-uv, axis=1)
    held = errors[1::2]
    result.update(xyz=xyz.tolist(), ray_angle_degrees=angle,
                  held_median_px=float(np.median(held)), held_p90_px=float(np.quantile(held, .9)),
                  min_depth=float(depths.min()))
    if depths.min() <= .2 or angle < min_angle:
        return {**result, 'reason': 'insufficient parallax or invalid cheirality'}
    if result['held_median_px'] <= median_limit and result['held_p90_px'] <= p90_limit:
        result.update(label='static_candidate', reason='fixed world point explains held observations')
    elif result['held_median_px'] > 2*median_limit or result['held_p90_px'] > 2*p90_limit:
        result.update(label='incompatible_fixed_point', reason='not explained by one fixed world point; not an instance/physical label')
    else:
        result['reason'] = 'ambiguous reprojection'
    return result


def region_support(labels, static_uv, incompatible_uv, minimum_seeds=2):
    """Generic image regions receive geometric seeds; unsupported pixels stay unknown."""
    height, width = labels.shape
    n = int(labels.max())+1
    counts = []
    for seeds in [static_uv, incompatible_uv]:
        seeds = np.asarray(seeds, dtype=float).reshape(-1, 2)
        xy = np.rint(seeds).astype(int)
        valid = (xy[:, 0] >= 0)&(xy[:, 0] < width)&(xy[:, 1] >= 0)&(xy[:, 1] < height)
        counts.append(np.bincount(labels[xy[valid, 1], xy[valid, 0]], minlength=n))
    accepted = (counts[0] >= minimum_seeds)&(counts[1] == 0)
    return accepted[labels]


def static_objective(rgb, alpha, target, positive):
    """Only image-geometric positives supervise; unknown is never empty background."""
    if not bool(positive.any()):
        raise ValueError('No inferred static support; preserve unknown')
    return (rgb[:, positive]-target[:, positive]).abs().mean()+.2*(alpha.squeeze(0)[positive]-1).square().mean()
