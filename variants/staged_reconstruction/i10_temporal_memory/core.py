"""Observed tracklet stitching and temporal bundle loss; no inferred target depth."""
from copy import deepcopy
from collections import defaultdict
import math
import numpy as np


def observations(track):
    return {track['start']+i: np.asarray(p, np.float64) for i, p in enumerate(track['uv'])}


def overlap(a, b, parity):
    left, right = observations(a), observations(b)
    frames = sorted(f for f in left.keys() & right.keys() if f % 2 == parity)
    error = np.array([np.linalg.norm(left[f]-right[f]) for f in frames])
    return dict(frames=frames, rms=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                maximum=float(error.max()) if len(error) else None)


def propose_links(tracks, config):
    """Fit-only mutual nearest matching; held coordinates cannot nominate links."""
    groups = defaultdict(list)
    for t in sorted(tracks, key=lambda t: t['id']):
        groups[t['start']].append(t)
    proposals, rejected = [], []
    for start, left in sorted(groups.items()):
        right = groups.get(start+config['next_start_delta'], [])
        if not right:
            continue
        costs = np.full((len(left), len(right)), np.inf)
        stats = {}
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                q = overlap(a, b, 0)
                if (len(q['frames']) >= config['min_overlap_fit'] and q['rms'] is not None
                        and np.isfinite(q['rms']) and q['maximum'] <= config['link_max_px']):
                    costs[i, j] = q['rms']; stats[i, j] = q
        for i, a in enumerate(left):
            if not np.isfinite(costs[i]).any():
                continue
            j = int(costs[i].argmin()); score = float(costs[i, j])
            if int(costs[:, j].argmin()) != i or score > config['link_rms_px']:
                continue
            row = np.sort(costs[i]); col = np.sort(costs[:, j])
            margin = min(row[1] if len(row)>1 else np.inf, col[1] if len(col)>1 else np.inf)-score
            q = dict(left=a['id'], right=right[j]['id'], fit=stats[i, j],
                     alternative_margin=None if not np.isfinite(margin) else float(margin))
            if margin < config['link_alternative_margin_px']:
                rejected.append(dict(q, reason='fit_ambiguous'))
            else:
                proposals.append(q)
    return sorted(proposals, key=lambda q: (q['fit']['rms'], q['left'], q['right'])), rejected


def build_bank(tracks, evidence, config):
    original = {t['id']: t for t in tracks}
    assert len(original) == len(tracks)
    proposed, rejected = propose_links(tracks, config)
    parent = {name: name for name in original}
    members = {name: [name] for name in original}
    def find(name):
        while parent[name] != name:
            name = parent[name]
        return name
    links = []
    for q in proposed:
        a, b = find(q['left']), find(q['right'])
        if a == b:
            continue
        joined = sorted(members[a]+members[b])
        starts = [original[name]['start'] for name in joined]
        bad = len(set(starts)) != len(starts)
        for i, name in enumerate(joined):
            for other in joined[i+1:]:
                check = overlap(original[name], original[other], 0)
                if check['maximum'] is not None and (not np.isfinite(check['maximum']) or check['maximum'] > config['link_max_px']):
                    bad = True
        if bad:
            rejected.append(dict(q, reason='fit_group_conflict')); continue
        # Membership has been fixed using fit. Held validates; it cannot choose an alternative.
        parent[b] = a; members[a] = joined; links.append(deepcopy(q))
    accepted_roots = {t['id']: t['binding']['row'] for t in evidence['rows'] if t['accepted']}
    memories = []
    for key in sorted(original):
        if find(key) != key:
            continue
        ids = members[key]; values = defaultdict(list); held_checks = []
        for name in ids:
            for f, p in observations(original[name]).items():
                values[f].append(p)
        for q in links:
            if q['left'] in ids and q['right'] in ids:
                held_checks.append(overlap(original[q['left']], original[q['right']], 1))
        held_ok = all(len(q['frames']) >= config['min_overlap_fit'] and q['rms'] is not None
                      and np.isfinite(q['rms']) and q['rms'] <= config['link_rms_px']
                      and q['maximum'] <= config['link_max_px'] for q in held_checks)
        # Check all duplicate held observations, including non-adjacent windows.
        for f, points in values.items():
            if f % 2 and len(points)>1:
                arr = np.array(points)
                if not np.isfinite(arr).all() or np.linalg.norm(arr[:, None]-arr[None], axis=-1).max() > config['link_max_px']:
                    held_ok = False
        frames = sorted(values)
        uv = [np.median(np.array(values[f]), axis=0).tolist() for f in frames]
        roots = sorted({accepted_roots[name] for name in ids if name in accepted_roots})
        candidate = len(ids)>=2 and len(frames)>=config['minimum_memory_span'] and held_ok and bool(roots)
        memories.append(dict(id=min(ids), track_ids=sorted(ids), frames=frames, uv=uv,
            nominated_rows=roots, candidate=bool(candidate), held_links_validated=bool(held_ok),
            overlap_observation_counts=[len(values[f]) for f in frames],
            physical_identity=False, no_gap_interpolation=True))
    return dict(memories=memories, fit_links=links, rejected_links=rejected,
                original_track_count=len(tracks), original_tracks_preserved=sorted(original))


def bind_memory(memory, xyz, projected, depth, config):
    frames = memory['frames']; uv = np.array(memory['uv']); rows = memory['nominated_rows']
    if not memory['candidate']:
        return dict(accepted=False, reason='not_long_observed_anchored_memory')
    fit = np.array([i for i, f in enumerate(frames) if not f % 2]); held = np.array([i for i, f in enumerate(frames) if f % 2])
    error = np.linalg.norm(projected[frames][:, rows]-uv[:, None], axis=-1)
    valid = np.isfinite(error).all(0) & np.isfinite(xyz[frames][:, rows]).all((0, 2)) & (depth[frames][:, rows] > config['depth_min']).all(0)
    scores = [np.sqrt(np.mean(error[s]**2, axis=0)) for s in [fit, held]]
    selected = [int(np.argmin(np.where(valid, s, np.inf))) for s in scores]
    if not valid.any():
        return dict(accepted=False, reason='nonfinite_or_nonpositive')
    i, j = selected; row = rows[i]; other = rows[j]
    alternatives = [k for k, r in enumerate(rows) if valid[k] and k != i and
        np.linalg.norm(xyz[frames[len(frames)//2], r]-xyz[frames[len(frames)//2], row]) >= config['binding_world_alternative_distance']]
    margin = min((float(scores[0][k]-scores[0][i]) for k in alternatives), default=None)
    separation = float(np.linalg.norm(xyz[frames, row]-xyz[frames, other], axis=-1).max())
    pixels = float(np.linalg.norm(projected[frames, row]-projected[frames, other], axis=-1).max())
    passed = (scores[0][i] <= config['projection_rms_max'] and scores[1][i] <= config['projection_rms_max']
        and error[:, i].max() <= config['projection_max'] and (margin is None or margin>=config['binding_margin_px'])
        and separation <= config['equivalent_world_max'] and pixels <= config['equivalent_projection_max']
        and scores[0][j] <= config['projection_rms_max'] and scores[1][j] <= config['projection_rms_max']
        and error[:, j].max() <= config['projection_max'])
    return dict(accepted=bool(passed), row=int(row), swapped_row=int(other), fit_rms=float(scores[0][i]),
        held_rms=float(scores[1][i]), maximum=float(error[:, i].max()), alternative_margin=margin,
        reciprocal_world=separation, reciprocal_pixels=pixels, parent_geometry_is_hypothesis=True)


def measurement(memory, projected, depth, visibility, appearance, config):
    error = np.linalg.norm(np.asarray(projected)-memory['uv'], axis=-1)
    frames = []
    for i, f in enumerate(memory['frames']):
        geom = bool(np.isfinite(error[i]) and np.isfinite(depth[i]) and depth[i] > config['depth_min'])
        app = appearance[i]; vis = visibility[i]
        known = bool(geom and vis['passed'] and app['observed_std'] >= config['texture_std_255_min']
                     and app['rendered_std'] >= config['texture_std_255_min'])
        passed = bool(known and error[i] <= config['projection_max'] and app['ncc'] >= config['ncc_min'])
        frames.append(dict(index=f, known=known, passed=passed, error=None if not np.isfinite(error[i]) else float(error[i]),
                           visibility=vis, appearance=app))
    splits = [[i for i, f in enumerate(memory['frames']) if f % 2 == parity] for parity in [0, 1]]
    count = [sum(frames[i]['passed'] for i in s) for s in splits]
    known = [sum(frames[i]['known'] for i in s) for s in splits]
    required = [max(config['minimum_valid_fit'] if parity == 0 else config['minimum_valid_held'],
                    math.ceil(len(s)*config['minimum_valid_fraction'])) for parity, s in enumerate(splits)]
    rms = [float(np.sqrt(np.mean(error[s]**2))) for s in splits]
    finite = np.isfinite(error).all() and np.isfinite(depth).all() and (np.asarray(depth) > config['depth_min']).all()
    passed = bool(finite and all(c>=n for c, n in zip(count, required)) and max(rms)<=config['projection_rms_max'] and error.max()<=config['projection_max'])
    kind = 'pass' if passed else ('contradiction' if all(c>=n for c, n in zip(known, required)) else 'unknown')
    confidence = min(c/len(s) for c, s in zip(count, splits))*max(0., 1-max(rms)/config['projection_max']) if finite else 0.
    return dict(passed=passed, kind=kind, confidence=float(confidence) if passed else 0., frames=frames,
                fit_count=count[0], held_count=count[1], required=required, fit_rms=rms[0], held_rms=rms[1])


def keyframes(frame, memories, states, config):
    options = set()
    for m, state in zip(memories, states):
        if state['status'] != 'active' or frame not in m['frames'] or frame % 2:
            continue
        passing = {q['index'] for q in state['measurement']['frames'] if q['passed']}
        if frame not in passing:
            continue
        options.update(f for f in m['frames'] if f != frame and not f % 2 and f in passing)
    return sorted(options, key=lambda f: (-abs(f-frame), f))[:config['memory_frame_budget']]


def add_memory(base, world_at, views, k, frame, memories, states, config, enabled=False):
    if not enabled:
        return base, base*0, dict(memories=0, endpoint_pairs=0, keyframes=[], denominator=0)
    import torch
    endpoints = keyframes(frame, memories, states, config)
    if not endpoints:
        return base, base*0, dict(memories=0, endpoint_pairs=0, keyframes=[], denominator=0)
    cache = {f: world_at(f) for f in [frame]+endpoints}
    def project(f, row):
        point = cache[f][row]; view = point.new_tensor(views[f]); intrinsic = point.new_tensor(k)
        camera = point@view[:3, :3].T+view[:3, 3]; pixel = camera@intrinsic.T
        return pixel[:2]/pixel[2].clamp_min(1e-8), camera[2]
    loss = base*0; applied=denominator=used=0
    for m, state in zip(memories, states):
        if frame not in m['frames'] or frame % 2:
            continue
        pairs = [f for f in endpoints if f in m['frames']]
        if not pairs:
            continue
        denominator += 1
        if state['status'] != 'active':
            continue
        passing = {q['index'] for q in state['measurement']['frames'] if q['passed']}
        if frame not in passing:
            continue
        targets = {f: p for f, p in zip(m['frames'], m['uv'])}; row = m['binding']['row']
        current, z = project(frame, row); target = current.new_tensor(targets[frame]); error = current-target
        if not bool(torch.isfinite(error).all() and z.detach()>config['depth_min'] and torch.linalg.vector_norm(error.detach())<=config['projection_max']):
            continue
        subtotal=base*0; n=0
        for other in pairs:
            if other not in passing:
                continue
            endpoint, depth = project(other, row); end_error=endpoint-endpoint.new_tensor(targets[other])
            if not bool(torch.isfinite(end_error).all() and depth.detach()>config['depth_min'] and torch.linalg.vector_norm(end_error.detach())<=config['projection_max']):
                continue
            robust = lambda e: torch.sqrt(1+e.square().sum())-1
            subtotal += (robust(error)+robust(end_error)+robust(end_error-error))/3
            n+=1
        if n:
            loss += state['confidence']*subtotal/max(len(pairs), 1); used+=1; applied+=n
    weighted=config['track_weight']*loss/max(denominator, 1)
    return base+weighted, weighted, dict(memories=used, endpoint_pairs=applied, keyframes=endpoints, denominator=denominator)
