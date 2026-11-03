"""Count-free train-image tracks; no model depth, component identity or re-ID."""
import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, binary_erosion, binary_dilation


def appearance_masks(rgba, erosion=2, suppress_highlights=False):
    """Appearance envelope only; no connected-component IDs or body counting."""
    r, g, b, alpha = np.asarray(rgba, np.float32).transpose(2, 0, 1)
    yellow_or_orange = (alpha > .95) & (r > .25) & (g > .05) & (b < .85*np.minimum(r, g))
    body = binary_fill_holes(binary_closing(yellow_or_orange, structure=np.ones((3, 3))))
    body &= alpha > .95
    visible = binary_erosion(body, iterations=erosion)
    if suppress_highlights:
        bright = (np.minimum(r, g) > .65) & (b > .5) & (b > .65*np.minimum(r, g))
        visible &= ~binary_dilation(bright, structure=np.ones((5, 5)))
    return body, visible


def sample(field, points):
    """Bilinear sampling at x/y pixel centers, without a hidden resize."""
    points = np.asarray(points, np.float32)
    return cv2.remap(np.asarray(field, np.float32), points[:, 0:1], points[:, 1:2],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).reshape(len(points), *field.shape[2:])


def patch_ncc(first, second, points, targets, radius=2):
    offsets = np.array([(x, y) for y in range(-radius, radius+1)
                        for x in range(-radius, radius+1)], np.float32)
    shape = (len(points), len(offsets))
    a = sample(first, (points[:, None]+offsets).reshape(-1, 2)).reshape(shape)
    b = sample(second, (targets[:, None]+offsets).reshape(-1, 2)).reshape(shape)
    a -= a.mean(1, keepdims=True)
    b -= b.mean(1, keepdims=True)
    denom = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
    score = np.divide((a*b).sum(1), denom, out=np.zeros(len(a), np.float32), where=denom > 1e-8)
    return score, np.sqrt((a*a).mean(1)), np.sqrt((b*b).mean(1))


def step(points, forward, backward, source_mask, target_mask, first, second, config):
    points = np.asarray(points, np.float32)
    h, w = forward.shape[:2]
    targets = points+sample(forward, points)
    returned = targets+sample(backward, targets)
    closure = np.linalg.norm(returned-points, axis=1)
    margin = config['patch_radius']+1
    inside = ((points[:, 0] >= margin) & (points[:, 0] <= w-1-margin)
              & (points[:, 1] >= margin) & (points[:, 1] <= h-1-margin)
              & (targets[:, 0] >= margin) & (targets[:, 0] <= w-1-margin)
              & (targets[:, 1] >= margin) & (targets[:, 1] <= h-1-margin))
    ncc, astd, bstd = patch_ncc(first, second, points, targets, config['patch_radius'])
    good = (inside & np.isfinite(targets).all(1) & (closure <= config['fb_max_px'])
            & (sample(source_mask, points) >= .99) & (sample(target_mask, targets) >= .99)
            & (ncc >= config['ncc_min']) & (astd >= config['patch_std_min'])
            & (bstd >= config['patch_std_min']))
    return targets, good, closure, ncc


def extract_window(start, ids, times, grays, masks, flows, config):
    corners = cv2.goodFeaturesToTrack(grays[start], maxCorners=config['max_corners'],
        qualityLevel=config['corner_quality'], minDistance=config['seed_distance'],
        mask=(masks[start]*255).astype(np.uint8), blockSize=3)
    if corners is None:
        return [], {'seeds': 0, 'short': 0, 'reverse_rejected': 0}
    initial = corners[:, 0, :].astype(np.float32)
    points = initial.copy()
    alive = np.ones(len(points), bool)
    lengths = np.ones(len(points), int)
    history, confidence = [points.copy()], [np.ones(len(points), np.float32)]
    for j in range(start, min(len(ids)-1, start+config['max_observations']-1)):
        points, good, fb, ncc = step(points, *flows[j], masks[j], masks[j+1],
                                     grays[j], grays[j+1], config)
        alive &= good
        lengths += alive
        history.append(points.copy())
        confidence.append(np.exp(-fb)*np.clip(ncc, 0, 1))
        if not alive.any():
            break
    history, confidence = np.asarray(history), np.asarray(confidence)
    tracks = []
    accepted = np.zeros(len(initial), bool)
    if config.get('check_valid_prefixes', False):
        candidates = range(int(lengths.max()), config['min_observations']-1, -1)
    else:
        candidates = sorted(set(lengths[lengths >= config['min_observations']]))
    for length in candidates:
        rows = np.flatnonzero((lengths >= length if config.get('check_valid_prefixes', False)
                               else lengths == length) & ~accepted)
        if not len(rows):
            continue
        returned = history[length-1, rows].copy()
        valid = np.ones(len(rows), bool)
        for j in range(start+length-2, start-1, -1):
            forward, backward = flows[j]
            returned, good, _, _ = step(returned, backward, forward, masks[j+1], masks[j],
                                         grays[j+1], grays[j], config)
            valid &= good
        closure = np.linalg.norm(returned-initial[rows], axis=1)
        valid &= closure <= config['fragment_closure_max_px']
        accepted[rows[valid]] = True
        for local in np.flatnonzero(valid):
            row = rows[local]
            track_times = [float(x) for x in times[start:start+length]]
            tracks.append({'source_index': start, 'source_frame': int(ids[start]),
                'frames': [int(x) for x in ids[start:start+length]], 'times': track_times,
                'uv_400': history[:length, row].tolist(),
                'confidence': (confidence[:length, row]*np.exp(-closure[local])).tolist(),
                'reverse_closure_px': float(closure[local]), 'bootstrap_count': 3,
                'time_band': min(2, int(track_times[0]*3)),
                'physical_material_ID_GT': False, 'occlusion_reidentified': False})
    return tracks, {'seeds': len(initial), 'short': int((lengths < config['min_observations']).sum()),
                    'reverse_rejected': int(((lengths >= config['min_observations']) & ~accepted).sum())}


def balanced_subset(tracks, maximum, seed):
    rng = np.random.default_rng(seed)
    groups = [[x for x in tracks if x['time_band'] == band] for band in range(3)]
    for group in groups:
        rng.shuffle(group)
    selected = []
    while len(selected) < maximum and any(groups):
        for group in groups:
            if group and len(selected) < maximum:
                selected.append(group.pop())
    for index, track in enumerate(selected):
        track['track_id'] = index
    return selected
