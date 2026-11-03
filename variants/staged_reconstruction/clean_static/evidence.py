"""Conditional ray evidence. Missing/occluded geometry is never free space."""
import numpy as np
from output.short_clip_diagnostic.camera_math import project


def ray_evidence(xyz, view, k, depth, reliable, relative=.025, absolute=.05):
    uv, z = project(xyz, view, k)
    height, width = depth.shape
    inside = np.isfinite(uv).all(1)&(z > .2)&(uv[:, 0] >= 2)&(uv[:, 0] < width-2)&(uv[:, 1] >= 2)&(uv[:, 1] < height-2)
    indices = np.flatnonzero(inside); xy = np.rint(uv[indices]).astype(int)
    sampled = np.zeros(len(xyz)); known = np.zeros(len(xyz), bool)
    sampled[indices] = depth[xy[:, 1], xy[:, 0]]
    known[indices] = reliable[xy[:, 1], xy[:, 0]] & (sampled[indices] > .2)
    tolerance = np.maximum(relative*np.maximum(z, 0), absolute)
    agree = known & (np.abs(z-sampled) <= tolerance)
    free = known & (z < sampled-tolerance)
    occluded = known & (z > sampled+tolerance)
    return agree, free, occluded, ~known


def classify(positive, free, protocol):
    """Quarantine is conditional rejection of static inclusion, not a dynamic ID."""
    fraction = free/np.maximum(positive+free, 1)
    return (free >= protocol['minimum_contradiction_views']) & (fraction >= protocol['contradiction_fraction'])


def accumulation(xyz, views, k, depths, reliable, protocol, frame_ids=None):
    positive = np.zeros(len(xyz), int); free = positive.copy(); occluded = positive.copy(); unknown = positive.copy()
    pos_views, free_views = [], []
    for view, depth, valid in zip(views, depths, reliable):
        a, b, c, d = ray_evidence(xyz, view, k, depth, valid, protocol['depth_relative_tolerance'], protocol['depth_absolute_tolerance'])
        positive += a; free += b; occluded += c; unknown += d
        pos_views.append(a); free_views.append(b)
    quarantine = classify(positive, free, protocol)
    if frame_ids is not None:
        ids = np.asarray(frame_ids)
        free_array = np.asarray(free_views)
        first = np.where(free_array, ids[:, None], 10**9).min(0)
        last = np.where(free_array, ids[:, None], -10**9).max(0)
        span = np.maximum(last-first, 0)
        quarantine &= span >= protocol['minimum_contradiction_time_span']
    else: span = np.zeros(len(xyz), int)
    return {'positive': positive, 'free': free, 'occluded': occluded, 'unknown': unknown,
            'quarantine': quarantine, 'free_time_span': span,
            'positive_views': np.asarray(pos_views).T, 'free_views': np.asarray(free_views).T}
