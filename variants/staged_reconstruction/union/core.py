"""Image union operations: no components, instance masks or group cardinality."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from ..recovery.coverage import project, chamfer, sample_rows


def dynamic_mask(rgba):
    r, g, b = rgba[..., :3].transpose(2, 0, 1)
    return (rgba[..., 3] > .95) & (r > .25) & (g > .18) & (b < .8*np.minimum(r, g))


def initialize_union(rgba, view, K, static, budget, seed):
    """Global pixel sampling and uncertain continuous depth; never cluster pixels."""
    pixels = np.argwhere(dynamic_mask(rgba))
    if not len(pixels):
        raise ValueError('No supported dynamic foreground in initialization frame')
    generator = np.random.default_rng(seed)
    pixels = pixels[generator.choice(len(pixels), budget, replace=budget > len(pixels))]
    y, x = pixels.T
    camera_static = static@view[:3, :3].T+view[:3, 3]
    front = camera_static[camera_static[:, 2] > 0]
    if len(front) < 12:
        raise ValueError('Insufficient front static support for uncertain depth prior')
    projected = front@K.T
    projected = projected[:, :2]/projected[:, 2:]
    _, neighbors = cKDTree(projected).query(np.c_[x, y], k=min(64, len(front)), workers=1)
    depth = front[neighbors, 2].min(-1)-.25
    if np.any(depth <= .12):
        raise ValueError('Invalid positive camera depth prior')
    rays = np.c_[x, y, np.ones(len(x))]@np.linalg.inv(K).T
    camera_points = rays*(depth+generator.uniform(-.12, .12, len(x)))[:, None]
    inverse = np.linalg.inv(view)
    world = camera_points@inverse[:3, :3].T+inverse[:3, 3]
    center = world.mean(0)
    return {'xyz': (world-center).astype(np.float32), 'rgb': rgba[y, x, :3].astype(np.float32),
            'center': center.astype(np.float32), 'spacing': np.asarray(float(np.median(depth)/K[0, 0]*1.8))}


def union_coverage(positions, camera, mask):
    uv, valid = project(positions, camera)
    target = torch.nonzero(mask.to(positions.device))[:, [1, 0]].to(positions)
    target = target*camera.image_width/mask.shape[1]
    return chamfer(sample_rows(uv[valid], 768), sample_rows(target, 768), camera.image_width, two_way=False)
