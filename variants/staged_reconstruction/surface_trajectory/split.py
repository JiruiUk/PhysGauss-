"""Leakage-aware grouping; image duplicates and common weighted supports stay together."""
from collections import defaultdict
import numpy as np


def group_tracks(tracks, bindings, config):
    ids = [t['track_id'] for t in tracks]
    parent = {i:i for i in ids}
    lookup = {b['track_id']:b for b in bindings}
    edges = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for index, first in enumerate(tracks):
        i = first['track_id']
        a = dict(zip(first['frames'], first['uv_400']))
        for second in tracks[:index]:
            j = second['track_id']
            b = dict(zip(second['frames'], second['uv_400']))
            common = sorted(set(a)&set(b))
            bootstrap = set(first['frames'][:3])&set(second['frames'][:3])
            reasons = []
            if bootstrap and min(np.linalg.norm(np.asarray(a[n])-b[n]) for n in bootstrap) < config['source_patch_diameter_px']:
                reasons.append('source_patch_overlap')
            if len(common) >= config['duplicate_minimum_common_frames']:
                distance = np.array([np.linalg.norm(np.asarray(a[n])-b[n]) for n in common])
                if np.median(distance) <= config['duplicate_median_px'] and distance.max() <= config['duplicate_maximum_px']:
                    reasons.append('duplicate_image_correspondence')
            if i in lookup and j in lookup:
                wa = dict(zip(lookup[i]['rows'], lookup[i]['weights']))
                wb = dict(zip(lookup[j]['rows'], lookup[j]['weights']))
                if sum(min(w, wb.get(r, 0.)) for r, w in wa.items()) >= config['weighted_support_overlap']:
                    reasons.append('weighted_support_overlap')
            if reasons:
                parent[find(i)] = find(j)
                edges.append({'first':i, 'second':j, 'reasons':reasons})
    groups = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    normalized = {min(values):sorted(values) for values in groups.values()}
    return normalized, edges


def split_groups(groups, bindings, seed=20261103, minimum_holdout_groups=20, minimum_training_tracks=30):
    """Fixed, count-based allocation. Never uses target errors/model performance."""
    accepted = {b['track_id']:b for b in bindings}
    groups = {g:[i for i in ids if i in accepted] for g, ids in groups.items()}
    groups = {g:ids for g, ids in groups.items() if ids}
    if len(groups) <= minimum_holdout_groups:
        return dict(admitted=False, reason='Insufficient disjoint groups for holdout and supervision', groups=groups)
    # Reserve >=20 holdout groups; among remaining capacity use large groups to
    # meet the predeclared >=30 training observations without leaking duplicates.
    rng = np.random.default_rng(seed)
    tie_order = list(groups)
    rng.shuffle(tie_order)
    order = sorted(tie_order, key=lambda g:-len(groups[g]))
    training_groups = []
    target = max(minimum_training_tracks, int(round(.8*len(bindings))))
    for group in order[:len(groups)-minimum_holdout_groups]:
        if sum(len(groups[g]) for g in training_groups) >= target:
            break
        training_groups.append(group)
    holdout_groups = [g for g in groups if g not in training_groups]
    train = sorted(i for g in training_groups for i in groups[g])
    holdout = sorted(i for g in holdout_groups for i in groups[g])
    admitted = len(train) >= minimum_training_tracks and len(holdout) >= 20 and len(holdout_groups) >= minimum_holdout_groups
    return dict(admitted=admitted, reason=None if admitted else 'Minimum supervised/holdout track counts not met',
                groups=groups, training_groups=training_groups, holdout_groups=holdout_groups,
                training_track_ids=train, holdout_track_ids=holdout,
                allocation_policy='Group-size allocation with fixed RNG tie-break; minimum holdout groups take priority over nominal80/20',
                actual_training_fraction=len(train)/len(bindings))
