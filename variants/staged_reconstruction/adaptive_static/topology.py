"""Evidence-gated birth; residuals alone cannot create geometry."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from output.short_clip_diagnostic.camera_math import w2c, intrinsic, project
from ..train_static import camera, render_static
from .geometry import depth_consistency
from .model import append_points


def select_births(xyz, current, score, support, ever_added, protocol, limit):
    distance = cKDTree(current).query(xyz)[0]
    valid = (support >= protocol['birth_min_support_views']) & np.isfinite(score)
    valid &= (score > 0) & (distance >= protocol['birth_min_distance'])
    valid[np.asarray(sorted(ever_added), dtype=int)] = False
    indices = np.flatnonzero(valid)
    order = indices[np.lexsort((indices, -score[indices]))]
    chosen = []
    for index in order:
        if chosen and np.linalg.norm(xyz[chosen]-xyz[index], axis=1).min() < protocol['birth_min_distance']:
            continue
        chosen.append(int(index))
        if len(chosen) >= limit: break
    return chosen if limit > 0 else []


@torch.no_grad()
def birth(renderer, model, optimizer, frames, meta, targets, geometry, reservoir, depths,
          protocol, step, ever_added, maximum):
    xyz = reservoir['xyz']; total = len(xyz)
    support = np.zeros(total, int); score_sum = np.zeros(total)
    k = intrinsic(meta['camera_angle_x'], 'metadata', 400, 225)
    for j, frame in enumerate(frames):
        name = str(int(frame['file_path'].replace('\\', '/').split('/')[-1]))
        view = w2c(frame['transform_matrix'], 'metadata_cv')
        uv, z = project(xyz, view, k)
        good, _ = depth_consistency(xyz, uv, depths[name], view, view, k,
            protocol['depth_relative_tolerance'], protocol['reprojection_limit_px'])
        good &= reservoir['support'][:, j] & (z > 0)
        xy = np.rint(np.nan_to_num(uv)).astype(int)
        inside = (xy[:, 0] >= 0)&(xy[:, 0] < 400)&(xy[:, 1] >= 0)&(xy[:, 1] < 225)
        indices = np.flatnonzero(good & inside)
        if not len(indices): continue
        x, y = xy[indices].T
        indices = indices[geometry[name][y, x] >= protocol['admission_threshold']]
        if not len(indices): continue
        x, y = xy[indices].T
        package = render_static(renderer, camera(frame, meta, 400), model)
        pred = package['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        alpha = package['alpha'].squeeze().cpu().numpy()
        residual = np.abs(pred-targets[name]).mean(2)[y, x]
        missing = alpha[y, x] < protocol['birth_missing_alpha_threshold']
        trigger = missing | (residual > protocol['birth_residual_threshold'])
        support[indices] += 1
        score_sum[indices] += np.where(trigger, residual+(1-alpha[y, x]).clip(0, 1), 0)
    score = score_sum/np.maximum(support, 1)
    count = min(maximum, protocol['max_static_points']-len(model.get_xyz))
    ids = select_births(xyz, model.get_xyz.detach().cpu().numpy(), score, support, ever_added, protocol, count)
    if ids:
        append_points(model, optimizer, xyz[ids], reservoir['rgb'][ids], ids, step, protocol['birth_opacity'])
        ever_added.update(ids)
    return {'step': step, 'operation': 'birth', 'ids': ids, 'points': len(model.get_xyz),
            'support_counts': support[ids].tolist(), 'scores': score[ids].tolist(),
            'geometry_required': True, 'ever_added_count': len(ever_added)}
