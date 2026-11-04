"""Independent-seed, recurring, spatially sampled color anchors; no identity GT."""
import numpy as np
import torch
from .core import recurring_candidates


def select_seed(support, good_mass_fraction, opacity, positions, *, floor=.25,
                fraction=.02, minimum_frames=5, voxel_size):
    support = torch.as_tensor(support, dtype=torch.float64)
    good = torch.as_tensor(good_mass_fraction, dtype=torch.float64)
    opacity = torch.as_tensor(opacity, dtype=torch.float64).reshape(-1)
    positions = np.asarray(positions)
    if good.shape != support.shape or positions.shape != (*support.shape, 3) or len(opacity) != support.shape[1]:
        raise ValueError('Expected matching TN support, quality and TN3 trajectories')
    if not np.isfinite(positions).all() or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('Finite positions and positive voxel size required')
    if not torch.isfinite(good).all() or not torch.isfinite(opacity).all():
        raise ValueError('Finite quality and opacity required')
    valid = (good >= .95) & (opacity[None] > .1) & (support >= floor)
    selected = recurring_candidates(support, valid, fraction=fraction, minimum_frames=minimum_frames)
    per_frame = selected['per_frame'].numpy()
    voxel_winners = np.zeros(support.shape[1], dtype=bool)
    score = support.numpy()
    for t, mask in enumerate(per_frame):
        ids = np.flatnonzero(mask)
        order = ids[np.argsort(-score[t, ids], kind='stable')]
        cells = np.floor(positions[t, order]/voxel_size).astype(np.int64)
        _, first = np.unique(cells, axis=0, return_index=True)
        voxel_winners[order[first]] = True
    period = selected['period'].numpy()
    return {'per_frame': per_frame, 'period': period, 'voxel_candidate_union': voxel_winners,
        'anchors': voxel_winners & (period >= minimum_frames)}


def compare_seeds(a, b):
    if a.shape != b.shape or a.ndim != 2 or a.dtype != bool or b.dtype != bool:
        raise ValueError('Expected two boolean TN selections')
    intersection = (a & b).sum(1)
    union = (a | b).sum(1)
    # No available anchors is failure, not perfect agreement between empty sets.
    return np.divide(intersection, union, out=np.zeros(len(a), dtype=np.float64), where=union > 0)
