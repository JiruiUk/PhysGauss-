"""Calibrated plane sweep, held-image checks and visibility-conditioned support."""
import numpy as np
import torch
import torch.nn.functional as F


def camera_center(views):
    c2w = np.linalg.inv(views)
    rays, origins = c2w[:, :3, 2], c2w[:, :3, 3]
    projectors = np.eye(3)[None]-rays[:, :, None]*rays[:, None, :]
    return np.linalg.solve(projectors.sum(0), np.einsum('nij,nj->i', projectors, origins))


def select_views(index, views, ids, count=6, min_angle=3., max_angle=32.):
    """Spread source camera angles and include both temporal directions if visible."""
    center = camera_center(views); origins = np.linalg.inv(views)[:, :3, 3]
    rays = origins-center; rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    angles = np.degrees(np.arccos(np.clip(rays@rays[index], -1, 1)))
    candidates = [j for j in range(len(ids)) if min_angle <= angles[j] <= max_angle and abs(ids[j]-ids[index]) >= 4]
    chosen = []
    for angle in [6., 12., 24.]:
        for direction in [-1, 1]:
            available = [j for j in candidates if j not in chosen and (ids[j]-ids[index])*direction > 0]
            if available:
                chosen.append(min(available, key=lambda j: (abs(angles[j]-angle), abs(ids[j]-ids[index]))))
    for j in sorted(candidates, key=lambda j: abs(angles[j]-16.)):
        if len(chosen) >= count: break
        if j not in chosen: chosen.append(j)
    if len(chosen) < count or max(ids[j] for j in chosen)-min(ids[j] for j in chosen) < 20:
        return []
    # Four fitting images, two held images with different camera angles.
    chosen = sorted(chosen[:count], key=lambda j: angles[j])
    return [chosen[j] for j in [0, 1, 3, 4, 2, 5]]


def patch_features(images, kernel=5):
    """Brightness/contrast normalized generic patches, not semantic colors."""
    gray = .299*images[:, 0:1]+.587*images[:, 1:2]+.114*images[:, 2:3]
    raw = F.unfold(gray, kernel, padding=kernel//2).reshape(len(images), kernel*kernel, *gray.shape[-2:])
    centered = raw-raw.mean(1, keepdim=True)
    norm = centered.square().sum(1, keepdim=True).sqrt()
    features = centered/norm.clamp_min(1e-5)
    valid = (norm[:, 0]/(kernel*kernel)**.5 >= .008)&(images.mean(1) > .025)
    return features, valid


def backproject(uv, depths, view, k):
    """Pixels[N,2], depths[D] -> world[D,N,3]."""
    rays = torch.cat([uv, torch.ones_like(uv[:, :1])], 1)@torch.linalg.inv(k).T
    local = rays[None]*depths[:, None, None]
    inverse = torch.linalg.inv(view)
    return local@inverse[:3, :3].T+inverse[:3, 3]


def project_torch(points, view, k, height, width):
    local = points@view[:3, :3].T+view[:3, 3]
    q = local@k.T
    uv = q[..., :2]/q[..., 2:].clamp_min(1e-6)
    grid = torch.stack([2*uv[..., 0]/(width-1)-1, 2*uv[..., 1]/(height-1)-1], -1)
    valid = (local[..., 2] > .2)&(uv[..., 0] >= 2)&(uv[..., 0] < width-2)&(uv[..., 1] >= 2)&(uv[..., 1] < height-2)
    return grid, valid, uv, local[..., 2]


def source_cost(points, reference, source, valid_source, view, k):
    height, width = source.shape[-2:]
    grid, good, _, _ = project_torch(points, view, k, height, width)
    sampled = F.grid_sample(source[None], grid[None], align_corners=True, padding_mode='zeros')[0]
    # Renormalize bilinearly interpolated feature vectors before comparing.
    sampled = sampled/sampled.square().sum(0, keepdim=True).sqrt().clamp_min(1e-5)
    texture = F.grid_sample(valid_source[None, None].float(), grid[None], mode='nearest', align_corners=True)[0, 0] > .5
    cost = ((1-(sampled*reference[:, None]).sum(0).clamp(-1, 1))/2).clamp(0, 1)
    return torch.where(good&texture, cost, torch.ones_like(cost))


@torch.no_grad()
def sweep(uv, features, texture, views, k, depths, reference_index, sources, margin=.012, cost_limit=.28):
    q = uv.long(); reference = features[reference_index, :, q[:, 1], q[:, 0]]
    points = backproject(uv, depths, views[reference_index], k)
    costs = torch.stack([source_cost(points, reference, features[j], texture[j], views[j], k) for j in sources[:4]])
    # At most one fitting view may be occluded or inconsistent.
    fit = costs.sort(0).values[:3].mean(0)
    best, best_index = fit.min(0)
    # Uniqueness must not change merely because the same depth range uses more
    # samples. Compare a separate depth hypothesis, not immediate grid neighbors.
    exclusion = (depths[:, None]-depths[best_index][None]).abs() <= .08*depths[best_index][None]
    alternative = fit.masked_fill(exclusion, float('inf')).min(0).values
    chosen = points[best_index, torch.arange(len(uv), device=uv.device)]
    held = torch.stack([source_cost(chosen[None], reference, features[j], texture[j], views[j], k)[0] for j in sources[4:]])
    good = (best <= cost_limit)&((alternative-best) >= margin)&(held.min(0).values <= cost_limit)
    return chosen, depths[best_index], good, best, alternative-best, held


def depth_consistency(xyz, ref_uv, source_depth, source_view, reference_view, k,
                      relative_tolerance=.025, reprojection_limit=1.5):
    """A missing/occluded depth is unknown, never a free-space vote."""
    from output.short_clip_diagnostic.camera_math import project
    uv, z = project(xyz, source_view, k)
    height, width = source_depth.shape
    good = (z > .2)&np.isfinite(uv).all(1)&(uv[:, 0] >= 1)&(uv[:, 0] < width-1)&(uv[:, 1] >= 1)&(uv[:, 1] < height-1)
    sampled = np.zeros(len(xyz)); rows = np.flatnonzero(good)
    xy = np.rint(uv[rows]).astype(int); sampled[rows] = source_depth[xy[:, 1], xy[:, 0]]
    good &= (sampled > 0)&(np.abs(sampled-z) <= relative_tolerance*np.maximum(z, 1e-6))
    local = np.c_[uv, np.ones(len(uv))]@np.linalg.inv(k).T*np.maximum(sampled, 1e-6)[:, None]
    inverse = np.linalg.inv(source_view)
    source_world = local@inverse[:3, :3].T+inverse[:3, 3]
    back, _ = project(source_world, reference_view, k)
    error = np.linalg.norm(back-ref_uv, axis=1)
    good &= np.isfinite(error)&(error <= reprojection_limit)
    return good, error


def fuse_voxels(xyz, rgb, supports, spacing):
    """Fuse observations of a surface voxel and preserve source visibility evidence."""
    keys = np.floor(xyz/spacing).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max())+1; weights = np.bincount(inverse)
    positions = np.stack([np.bincount(inverse, weights=xyz[:, j], minlength=count)/weights for j in range(3)], 1)
    colors = np.stack([np.bincount(inverse, weights=rgb[:, j], minlength=count)/weights for j in range(3)], 1)
    evidence = np.zeros((count, supports.shape[1]), bool)
    np.logical_or.at(evidence, inverse, supports)
    return positions.astype(np.float32), colors.astype(np.float32), evidence


def fit_plane(points, minimum=12, rms_limit=.025, second_min=.002, ratio_max=.08):
    if len(points) < minimum: return None
    fit, held = points[::2], points[1::2]
    center = fit.mean(0); _, singular, vectors = np.linalg.svd(fit-center, full_matrices=False)
    variances = singular**2/max(len(fit)-1, 1)
    if variances[1] < second_min or variances[2]/max(variances[1], 1e-12) > ratio_max: return None
    normal = vectors[-1]; error = np.abs((held-center)@normal)
    rms = float(np.sqrt(np.mean(error**2)))
    if rms > rms_limit or np.quantile(error, .9) > 2*rms_limit: return None
    return {'center': center, 'normal': normal, 'held_rms': rms, 'second_variance': float(variances[1])}


def plane_intersections(uv, plane, view, k):
    inverse = np.linalg.inv(view); origin = inverse[:3, 3]
    directions = (np.c_[uv, np.ones(len(uv))]@np.linalg.inv(k).T)@inverse[:3, :3].T
    denominator = directions@plane['normal']
    depth = (plane['center']-origin)@plane['normal']/np.where(np.abs(denominator) > 1e-6, denominator, 1e-6)
    xyz = origin+directions*depth[:, None]
    return xyz, depth, np.abs(denominator) > .05
