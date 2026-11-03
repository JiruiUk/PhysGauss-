"""Independent geometric admission and revisable appearance confidence."""
import numpy as np


def mutual_matches(first, second, ratio=.75):
    import cv2
    if first is None or second is None or min(len(first), len(second)) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = {m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(first, second, k=2)
               if m.distance < ratio*n.distance}
    reverse = {m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(second, first, k=2)
               if m.distance < ratio*n.distance}
    return [(i, j) for i, j in forward.items() if reverse.get(j) == i]


def epipolar_errors(uv0, uv1, view0, view1, k):
    relative = view1@np.linalg.inv(view0)
    tx, ty, tz = relative[:3, 3]
    skew = np.array([[0., -tz, ty], [tz, 0., -tx], [-ty, tx, 0.]])
    f = np.linalg.inv(k).T@skew@relative[:3, :3]@np.linalg.inv(k)
    x = np.c_[uv0, np.ones(len(uv0))]; y = np.c_[uv1, np.ones(len(uv1))]
    line1 = x@f.T; line0 = y@f
    numerator = np.abs((y*line1).sum(1))
    return numerator/np.maximum(np.minimum(np.linalg.norm(line1[:, :2], axis=1),
                                          np.linalg.norm(line0[:, :2], axis=1)), 1e-12)


def join_tracks(edges):
    """Never merge components containing distinct features in the same image."""
    parents, observations = {}, {}
    def root(node):
        if node not in parents:
            parents[node] = node; observations[node] = {node[0]: node[1]}
        while parents[node] != node:
            parents[node] = parents[parents[node]]; node = parents[node]
        return node
    conflicts = 0
    for a, b in edges:
        a, b = root(tuple(a)), root(tuple(b))
        if a == b:
            continue
        if set(observations[a]) & set(observations[b]):
            conflicts += 1; continue
        if len(observations[a]) < len(observations[b]):
            a, b = b, a
        parents[b] = a; observations[a].update(observations.pop(b))
    return [sorted(obs.items()) for obs in observations.values()], conflicts


def confidence_step(previous, geometry, photo, visible, rate=.5):
    """Rendered agreement cannot admit any pixel without independent geometry.

    Absent alpha is missing model support, not evidence of empty background.
    Geometry is a conditional upper bound, not a calibrated probability.
    """
    appearance = np.where(visible, .4+.6*np.clip(photo, 0, 1), .75)
    target = geometry*appearance
    return np.minimum(geometry, (1-rate)*previous+rate*target).astype(np.float32)


def weighted_objective(rgb, alpha, target, confidence, threshold=.5):
    import torch
    weight = torch.where(confidence >= threshold, confidence, torch.zeros_like(confidence))
    denominator = weight.sum().clamp_min(1)
    error = (rgb-target).abs().mean(0)+.2*(alpha.squeeze(0)-1).square()
    return (error*weight).sum()/denominator


def summary(confidence, threshold=.5):
    counts = {n: int((v >= threshold).sum()) for n, v in confidence.items()}
    return {'positive_pixels': sum(counts.values()), 'frames_any': sum(v > 0 for v in counts.values()),
            'frames_64': sum(v >= 64 for v in counts.values()), 'pixels_per_frame': counts,
            'fraction_all_image_pixels': sum(counts.values())/sum(v.size for v in confidence.values()),
            'fraction_is_not_static_recall': True}
