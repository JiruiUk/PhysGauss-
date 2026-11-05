"""Train-only binding checks; visibility is not a semantic/static label."""
from copy import deepcopy
import numpy as np
from ..i10_tracking.core import bind_track, sample


def reciprocal(track, xyz, predicted, depth, original, config):
    swapped = deepcopy(original)
    swapped['fit_observations'], swapped['held_observations'] = (
        original['held_observations'], original['fit_observations'])
    result = bind_track(track['uv'], xyz, predicted, depth, swapped)
    row = track['binding']['row']; other = result.get('row')
    world = projection = None
    if other is not None:
        world = float(np.linalg.norm(xyz[:, row]-xyz[:, other], axis=-1).max())
        projection = float(np.linalg.norm(predicted[:, row]-predicted[:, other], axis=-1).max())
    equivalent = (world is not None and np.isfinite([world, projection]).all()
                  and world <= config['equivalent_max_world_distance']
                  and projection <= config['equivalent_max_projection_distance_px'])
    return dict(passed=bool(result['accepted'] and equivalent),
                swapped_binding=result, same_persistent_ID=other == row,
                max_world_separation=world, max_projected_separation_px=projection,
                surface_equivalence_is_model_conditioned=True)


def patch_visibility(weight, alpha, uv, config):
    weight = np.asarray(weight, np.float32); alpha = np.asarray(alpha, np.float32)
    if weight.shape != alpha.shape or weight.ndim != 2:
        raise ValueError('Aligned full-scene compositing fields required')
    if not np.isfinite(weight).all() or not np.isfinite(alpha).all():
        raise ValueError('Finite fields required')
    if (weight < -1e-6).any() or (weight-alpha > 1e-5).any():
        raise ValueError('Expected pre-transmittance compositing contribution')
    radius = config['visibility_patch_radius']
    x, y = np.asarray(uv, np.float32)
    h, w = alpha.shape
    if not np.isfinite([x, y]).all() or x-radius < 0 or y-radius < 0 or x+radius > w-1 or y+radius > h-1:
        return dict(share=None, known=False, passed=False, reason='patch_outside_image')
    offsets = np.array([(i, j) for j in range(-radius, radius+1)
                        for i in range(-radius, radius+1)], np.float32)
    points = np.array([x, y])+offsets
    numerator = float(sample(np.maximum(weight, 0), points).astype(np.float64).sum())
    denominator = float(sample(alpha, points).astype(np.float64).sum())
    known = denominator > config['visibility_min_denominator']
    share = numerator/denominator if known else None
    return dict(share=share, known=known,
                passed=bool(known and share >= config['visibility_min_share']),
                numerator=numerator, denominator=denominator)


def visibility_gate(observations, config):
    good = np.array([o['passed'] for o in observations], bool)
    return dict(passed=bool(good.sum() >= config['visibility_min_frames'] and
                           good[config['held_observations']].sum() >= config['visibility_min_held_frames']),
                visible_frames=int(good.sum()),
                visible_held_frames=int(good[config['held_observations']].sum()),
                unknown_frames=sum(not o['known'] for o in observations))


def neighborhood(xyz, anchor, config):
    """A short-trajectory local proposal, not physical identity or a GT surface."""
    xyz = np.asarray(xyz, np.float64)
    if xyz.ndim != 3 or not np.isfinite(xyz).all():
        raise ValueError('Finite TN3 parent trajectories required')
    relative = xyz-xyz[:, anchor:anchor+1]
    distance = np.linalg.norm(relative, axis=-1)
    mid = len(xyz)//2
    ids = np.arange(xyz.shape[1])
    ids = ids[(ids != anchor) & (distance[mid] <= config['neighbor_radius'])]
    ids = ids[np.lexsort((ids, distance[mid, ids]))][:config['neighbor_limit']]
    rms = np.sqrt(np.mean(np.sum((relative-relative[mid:mid+1])**2, axis=-1), axis=0))
    valid = ((distance[:, ids].max(0) <= config['neighbor_radius']) &
             (rms[ids] <= config['relative_displacement_rms_max']))
    kept = ids[valid]
    return dict(qualified=bool(len(kept) >= config['minimum_neighbors']),
                anchor=int(anchor), candidate_rows=ids.tolist(), rows=kept.tolist(),
                max_distance=distance[:, kept].max(0).tolist(), relative_displacement_rms=rms[kept].tolist(),
                parent_shape_is_model_conditioned=True, physical_identity=False)
