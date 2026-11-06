"""Overlapping local surface hypotheses; all parent geometry remains a proxy."""
from collections import Counter
import numpy as np
from ..i10_revocable_surface.core import geometry, patch_measurement, transition, edge_measurement


def build_patches(evidence, xyz, projected, config):
    """Only fit frames nominate members. Held frames cannot affect nomination."""
    fit = config['fit']
    patches, options = [], []
    for t in sorted(evidence['rows'], key=lambda q: q['id']):
        if not t['accepted']:
            continue
        row = t['binding']['row']; start = t['start']
        x = np.asarray(xyz[start:start+12], np.float64)
        uv = np.asarray(projected[start:start+12], np.float64)
        index = len(patches)
        patches.append(dict(id=t['id'], start=start, band=t['band'], rows=[row], weights=[1.],
            uv=t['uv'], parent_centers=x[:, row].tolist(), is_anchor=True,
            owner=None, members=[], membership_weight=1., physical_identity=False))
        relative = x[fit] - x[fit, row:row+1]
        distance = np.linalg.norm(relative, axis=-1)
        mean = distance.mean(0)
        rms = np.sqrt(np.mean(np.sum((relative-relative.mean(0))**2, axis=-1), axis=0))
        valid = (np.isfinite(relative).all((0, 2)) &
            (distance.min(0) >= config['edge_min_distance']) &
            (distance.max(0) <= config['neighbor_radius']) &
            (np.ptp(distance, axis=0)/np.maximum(mean, 1e-12) <= config['edge_relative_length_range_max']) &
            (rms <= config['neighbor_relative_displacement_rms']))
        ids = np.flatnonzero(valid)
        ids = ids[np.lexsort((ids, mean[ids]))][:config['neighbor_limit']]
        for neighbor in ids:
            options.append((float(mean[neighbor]), int(neighbor), index))
    # Several plausible owners remain overlapping hypotheses, not hard segmentation.
    used = Counter(); kept = []
    for score, row, owner in sorted(options):
        if used[row] >= config['max_anchor_hypotheses_per_row']:
            continue
        used[row] += 1; kept.append((owner, row, score))
    for owner, row, score in sorted(kept):
        q = patches[owner]; start = q['start']; anchor = q['rows'][0]
        x = np.asarray(xyz[start:start+12], np.float64)
        p = np.asarray(projected[start:start+12], np.float64)
        target = np.asarray(q['uv']) + p[:, row] - p[:, anchor]
        q['members'].append(len(patches))
        patches.append(dict(id=q['id']+'/r'+str(row), start=start, band=q['band'], rows=[row], weights=[1.],
            uv=target.tolist(), parent_centers=x[:, row].tolist(), is_anchor=False, owner=owner,
            membership_weight=1./used[row], fit_distance=score, physical_identity=False,
            target_is_parent_transported_proxy=True, receives_track_auxiliary=False))
    return patches


def build_edges(patches, config):
    edges = []
    for owner, root in enumerate(patches):
        if not root['is_anchor']:
            continue
        members = root['members']; pairs = set()
        def propose(i, j):
            if i == j or patches[i]['rows'][0] == patches[j]['rows'][0]:
                return None
            lengths = np.linalg.norm(np.asarray(patches[i]['parent_centers'])-patches[j]['parent_centers'], axis=-1)
            if geometry(lengths, config['fit'], config):
                return lengths
            return None
        for i in members:
            if propose(owner, i) is not None:
                pairs.add(tuple(sorted((owner, i))))
            near = []
            for j in members:
                lengths = propose(i, j)
                if lengths is not None:
                    near.append((float(lengths[config['fit']].mean()), j))
            for _, j in sorted(near)[:config['triangle_neighbors']]:
                pairs.add(tuple(sorted((i, j))))
        for i, j in sorted(pairs):
            lengths = propose(i, j)
            edges.append(dict(i=i, j=j, owner=owner, start=root['start'], fit_selected=True,
                parent_lengths=lengths.tolist(), weight=min(patches[i]['membership_weight'], patches[j]['membership_weight'])))
    return edges


def add_constraints(base, world, view, k, frame, patches, edges, states, edge_states, config, enabled=False):
    if not enabled:
        return base, base*0, base*0, dict(tracks=0, edges=0, track_denominator=0, edge_denominator=0)
    import torch
    centers = torch.stack([world[q['rows'][0]] for q in patches])
    camera = centers@view[:3, :3].T+view[:3, 3]
    projected = camera@k.T
    uv = projected[:, :2]/projected[:, 2:3].clamp_min(1e-8)
    track, local = base*0, base*0
    nt = ne = dt = de = 0; usable = {}
    for i, (patch, state) in enumerate(zip(patches, states)):
        t = frame-patch['start']
        if t not in config['fit']:
            continue
        if patch['is_anchor']:
            dt += 1
        if state['status'] != 'active' or not state['measurement']['frames'][t]['passed']:
            continue
        error = uv[i]-uv.new_tensor(patch['uv'][t])
        if not bool(torch.isfinite(error).all() and camera[i, 2].detach()>config['depth_min'] and
                    torch.linalg.vector_norm(error.detach())<=config['projection_max']):
            continue
        usable[i] = state['confidence']
        if patch['is_anchor']:
            nt += 1
            track = track+state['confidence']*(torch.sqrt(1+error.square().sum())-1)
    for edge, state in zip(edges, edge_states):
        t = frame-edge['start']
        if t not in config['fit']:
            continue
        de += 1; i, j = edge['i'], edge['j']
        if state['status'] != 'active' or any(v not in usable for v in [i, j, edge['owner']]):
            continue
        length = torch.linalg.vector_norm(centers[j]-centers[i])
        if not bool(torch.isfinite(length) and config['edge_min_distance']<=length.detach()<=config['edge_max_distance']):
            continue
        confidence = min(state['confidence'], usable[i], usable[j], usable[edge['owner']])
        error = (length-edge['parent_lengths'][t])/config['local_scale']
        local = local+edge['weight']*confidence*(torch.sqrt(1+error.square())-1); ne += 1
    track = config['track_weight']*track/max(dt, 1)
    local = config['local_weight']*local/max(de, 1)
    return base+track+local, track, local, dict(tracks=nt, edges=ne, track_denominator=dt, edge_denominator=de)
