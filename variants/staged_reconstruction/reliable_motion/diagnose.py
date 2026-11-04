"""Attribute failed coverage to observation, recurrence and separation stages."""
import json
import numpy as np
from scipy.spatial import cKDTree
from ..prepare_data import ROOT, read, write, sha
from .prepare import EXP, WORK, ANCHORS


def stages(a, trusted, construction):
    x = a['positions']; score = a['per_seed_support'].mean(0)
    quality = (a['converged_mass_fraction'] >= .95) & (score > 1e-8)
    active = a['opacity'] > .1; keys = a['anchor_ids']; n = len(active)
    radius = construction['radius']; minimum = construction['minimum_sources']
    proposals = []; encoded = []
    for t in range(len(x)):
        target = np.flatnonzero(quality[t] & active); source = keys[trusted[t, keys]]
        if len(source) < minimum or not len(target):
            continue
        d, j = cKDTree(x[t, source]).query(x[t, target], k=min(construction['candidates'], len(source)))
        ids = source[j]; valid = (d <= radius) & (ids != target[:, None])
        code = target[:, None]*n+ids
        encoded.append(code[valid]); proposals.append((target, valid, code))
    codes = np.unique(np.concatenate(encoded)); recurrence = np.zeros(len(codes), bool)
    separation = np.zeros(len(codes), bool); counts = np.zeros(len(codes), np.int16)
    max_distance = np.zeros(len(codes), np.float32)
    for start in range(0, len(codes), 4096):
        ids = codes[start:start+4096]; i, j = ids//n, ids % n
        common = quality[:, i] & trusted[:, j]
        count = common.sum(0); distance = np.linalg.norm(x[:, i]-x[:, j], axis=-1)
        maximum = np.where(common, distance, 0).max(0)
        sl = slice(start, start+len(ids)); counts[sl] = count; max_distance[sl] = maximum
        recurrence[sl] = count >= construction['common_min']; separation[sl] = maximum <= radius
    masks = np.zeros((3, n), bool)
    for target, valid, code in proposals:
        loc = np.searchsorted(codes, code.clip(codes[0], codes[-1]))
        present = valid & (codes[loc] == code)
        for row, passed in enumerate([present, present & recurrence[loc], present & recurrence[loc] & separation[loc]]):
            masks[row, target[(passed.sum(1) >= minimum)]] = True
    report = {'currently_observed_anchor_count_quantiles': np.quantile(trusted[:, keys].sum(1), [0,.1,.5,1]).tolist(),
        'active_with_quality_at_least_once': int((quality.any(0) & active).sum()),
        'targets_with_three_reference_neighbors': int(masks[0].sum()),
        'targets_after_three_common_observations': int(masks[1].sum()),
        'targets_after_all_common_time_radius_before_rank_test': int(masks[2].sum()),
        'candidate_pairs': len(codes), 'pairs_with_three_common_observations': int(recurrence.sum()),
        'pairs_with_common_observations_and_all_time_radius': int((recurrence & separation).sum()),
        'maximum_common_distance_quantiles_recurring_pairs': np.quantile(max_distance[recurrence], [0,.1,.5,.9,1]).tolist()}
    return report, masks


def main():
    assert not (EXP/'coverage_diagnosis.json').exists(), 'Preserve failure diagnosis'
    plan = read(EXP/'graph_protocol.json')
    for name, expected in plan['inputs'].items():
        assert sha(ROOT/name) == expected, name
    a = dict(np.load(ANCHORS))
    observed = ((a['per_seed_support'] >= .25).all(0)
                & (a['converged_mass_fraction'] >= .95) & (a['opacity'][None] > .1))
    reports = {}; arrays = {}
    for name, trusted in [('rank_top2', a['trusted']), ('observation_qualified', observed)]:
        reports[name], arrays[name] = stages(a, trusted, plan['construction'])
    np.savez_compressed(WORK/'coverage_diagnosis.npz', **arrays)
    write(EXP/'coverage_diagnosis.json', {'definitions': reports,
        'active_parent_points': int((a['opacity'] > .1).sum()),
        'new_anchor_observations_outside_top2': int((observed[:, a['anchor_ids']] & ~a['trusted'][:, a['anchor_ids']]).sum()),
        'script_sha256': sha(__file__), 'input_sha256': sha(ANCHORS),
        'graph_protocol_sha256': sha(EXP/'graph_protocol.json'),
        'arrays_sha256': sha(WORK/'coverage_diagnosis.npz'),
        'scope': 'Exploratory attribution; no new graph qualification or training',
        'optimizer_updates': 0, 'test_RGB_read': False})
    print('RELIABLE_COVERAGE_DIAGNOSIS', json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
