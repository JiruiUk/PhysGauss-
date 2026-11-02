"""Independent ledger/point arithmetic and protected-input verification."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy import ndimage
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from output.surface_support.core import propagate
from variants.surface_correspondence.model import qualify_fragments
from variants.flow_comparison.geometry import sample, forward_backward_valid

REPORT = ROOT / 'experiments/object_membership'
OUT = ROOT / 'output/object_membership'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def arrays(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as s:
        for block in iter(lambda: s.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    locks = read(REPORT / 'input_lock.json')
    for name, expected in locks.items():
        assert digest(ROOT / name) == expected, name
    r = read(REPORT / 'results.json')
    assert digest(OUT / 'membership.npz') == r['arrays_sha256']
    z = arrays(OUT / 'membership.npz')
    assert str(z['checkpoint_sha256']) == r['checkpoint_sha256']
    assert int(z['protocol_version']) == r['protocol_version'] == 1
    plan = read(REPORT / 'plan.json')
    tracks = arrays(ROOT / 'output/surface_support/image_tracks.npz')
    components = {int(n): c for n, c in zip(tracks['frames'], tracks['components'])}
    flow = ROOT / 'data/derived/flow_comparison_train_only'
    manifest = read(flow / 'manifest.json')
    lookup = {(a['source'], a['target']): a for a in manifest['pairs']}
    cache = ROOT / 'runs/pidg_short_clip_diagnostic/cache'
    all_numbers = {n for ff in plan['domains'].values() for n in ff}
    geometry = {n: arrays(cache / f'frame{n:04d}.npz') for n in all_numbers}
    native = {n: arrays(cache / f'native{n:04d}.npz') for n in all_numbers}
    pair = {}
    for ff in plan['domains'].values():
        for a, b in zip(ff[:-1], ff[1:]):
            raw = arrays(flow / lookup[a, b]['file'])
            pair[a, b] = tuple(torch.tensor(raw[k], dtype=torch.float32) for k in ['forward', 'backward'])

    def point_inside(number, component):
        g = geometry[number]
        p, depth, cov = g['means'], g['depth'], g['cov']
        x, y = np.rint(p).astype(np.int64).T
        valid = (depth > .2) & np.isfinite(p).all(-1) & np.isfinite(cov).all((1, 2))
        valid &= (cov[:, 0, 0] > 0) & (cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0] > 0)
        valid &= (x >= 0) & (x < 400) & (y >= 0) & (y < 225)
        mask = ndimage.binary_erosion(components[number] == component, iterations=1)
        valid[valid] &= mask[y[valid], x[valid]]
        return valid

    def instant(number, component):
        mass = native[number]['mass']
        return (mass[:, component] >= .02) & (mass[:, 5] > 0) & (mass[:, component] / np.maximum(mass[:, 5], 1e-20) >= .95) & point_inside(number, component)

    def mass_fraction(number, labels, mapping):
        own = sum(float(native[number]['mass'][labels == seed, region].sum(dtype=np.float64)) for seed, region in enumerate(mapping) if region >= 0)
        return own / float(native[number]['alpha'][components[number] >= 0].sum(dtype=np.float64))

    checked = []
    for domain_row in r['domains']:
        name = domain_row['domain']
        ff = plan['domains'][name]
        obs = z[name + '_observed']
        mappings = z[name + '_mapping']
        # Recompute direction-specific visible image links for every prefix.
        for t, n in enumerate(ff):
            direction_maps = []
            for reverse in [False, True]:
                seq = list(reversed(ff[:t + 1])) if reverse else ff[:t + 1]
                labels = components[seq[0]].copy()
                alive = np.ones(4, bool)
                mapping, alive = qualify_fragments(labels, components[seq[0]], alive)
                for a, b in zip(seq[:-1], seq[1:]):
                    f, back = pair[min(a, b), max(a, b)]
                    labels, _ = propagate(labels, components[b], *( (back, f) if reverse else (f, back) ))
                    mapping, alive = qualify_fragments(labels, components[b], alive)
                    for seed in range(4):
                        if not alive[seed]:
                            labels[labels == seed] = -1
                direction_maps.append(mapping)
            fwd, rev = direction_maps
            expected = np.full(4, -1)
            for seed in range(4):
                candidates = [region for region in range(4) if fwd[seed] == region and rev[region] == seed and (fwd == region).sum() == 1 and (rev == seed).sum() == 1]
                if len(candidates) == 1:
                    expected[seed] = candidates[0]
            assert np.array_equal(expected, mappings[t])
            assert fwd.tolist() == domain_row['links'][t]['forward_seed_to_component']
            assert rev.tolist() == domain_row['links'][t]['reverse_component_to_seed']
            supported = np.full(obs.shape[1], -1, np.int16)
            proposals = []
            if t:
                p = torch.tensor(geometry[ff[t - 1]]['means'])
                q = torch.tensor(geometry[n]['means'])
                f, back = pair[ff[t - 1], n]
                error = torch.linalg.vector_norm(q - p - sample(f, p[:, None])[:, 0], dim=-1).numpy()
                valid = (sample(forward_backward_valid(f, back).float(), p[:, None])[:, 0] > .99).numpy()
                valid &= (p[:, 0].numpy() >= 0) & (p[:, 0].numpy() <= 399) & (p[:, 1].numpy() >= 0) & (p[:, 1].numpy() <= 224)
            for seed, region in enumerate(expected):
                good = np.zeros(obs.shape[1], bool)
                if region >= 0:
                    good = instant(n, region)
                    if t:
                        previous_region = mappings[t - 1, seed]
                        good &= instant(ff[t - 1], previous_region) if previous_region >= 0 else False
                        good &= valid & np.isfinite(error) & (error <= 2)
                proposals.append(good)
            unique = np.sum(proposals, axis=0) == 1
            supported[unique] = np.argmax(proposals, axis=0)[unique]
            assert np.array_equal(supported, obs[t]), (name, n)
            # Independent prefix oracle: sets of ever-seen IDs and any triple.
            prefix = obs[:t + 1]
            seen = np.stack([(prefix == k).any(axis=0) for k in range(4)])
            conflict = seen.sum(axis=0) > 1
            qualified = np.zeros(obs.shape[1], bool)
            for j in range(2, t + 1):
                qualified |= (obs[j] >= 0) & (obs[j] == obs[j - 1]) & (obs[j] == obs[j - 2])
            history = np.where(qualified & ~conflict, seen.argmax(axis=0), -1)
            visible = np.where((supported >= 0) & (history == supported), supported, -1)
            assert np.array_equal(history, z[name + '_historical_owner'][t])
            assert np.array_equal(visible, z[name + '_visible_owner'][t])
            assert np.array_equal(conflict, z[name + '_conflict'][t])
            row = domain_row['frames'][t]
            assert int((history >= 0).sum()) == row['historical_member_points']
            assert int((visible >= 0).sum()) == row['currently_visible_member_points']
            assert int((supported >= 0).sum()) == row['supported_points']
            assert int(conflict.sum()) == row['latched_conflict_points']
            assert [int((history == k).sum()) for k in range(4)] == row['historical_member_points_per_object']
            assert abs(mass_fraction(n, visible, expected) - row['current_visible_member_alpha_coverage']) < 1e-12
            assert abs(mass_fraction(n, supported, expected) - row['instantaneous_supported_alpha_coverage']) < 1e-12
            # Complete contribution conservation, using the native1600 caches.
            for k in range(4):
                actual = native[n]['mass'][:, k].sum(dtype=np.float64)
                expected_mass = native[n]['alpha'][components[n] == k].sum(dtype=np.float64)
                assert abs(actual - expected_mass) / max(expected_mass, 1) < 5e-4
        assert int((obs >= 0).any(axis=0).sum()) == domain_row['unique_ever_supported_points']
        assert np.flatnonzero(z[name + '_conflict'][-1]).tolist() == domain_row['final_conflict_row_ids']
        checked.append({'domain': name, 'observations': len(ff), 'independent_point_and_prefix_ledger_recompute': True})

    old = read(ROOT / 'experiments/short_clip_diagnostic/resolution_control.json')
    banks = {b['name']: arrays(ROOT / b['path'])['kernel_ids'] for b in old['native_seed_banks'] if b['qualified']}
    model = read(ROOT / 'experiments/short_clip_diagnostic/model_results.json')
    ff = plan['domains']['early_2_27']
    mapping = z['early_2_27_mapping']
    for row in r['patch_bridges']:
        candidate = []
        for n in row['seed_frames']:
            region = next(x['region'] for x in model['bank_frames'] if x['bank'] == row['new_bank'] and x['frame'] == n)
            ids = np.flatnonzero(mapping[ff.index(n)] == region)
            candidate.append(int(ids[0]) if len(ids) == 1 else -1)
        owner = candidate[0] if min(candidate) >= 0 and len(set(candidate)) == 1 else -1
        assert owner == row['early_local_object'] and candidate == row['verified_object_at_each_seed']
        current = set(banks[row['new_bank']].tolist())
        counts = [len(current & set(banks[f'A2_F{k}'].tolist())) for k in range(4)]
        assert counts == row['row_overlap_with_A2_objects']
        if owner >= 0:
            assert len(current - set(banks[f'A2_F{owner}'].tolist())) == row['rows_new_to_same_object_A2_bank']
    for row in r['causal_bank_comparison_after_A15_seeds']:
        n = row['frame']
        t = ff.index(n)
        assert n > 19
        for mode in ['A2_only', 'A2_plus_A15']:
            expected = np.full(z['early_2_27_observed'].shape[1], -1, np.int16)
            for k in range(4):
                ids = set(banks[f'A2_F{k}'].tolist())
                if mode == 'A2_plus_A15':
                    for b in r['patch_bridges']:
                        if b['early_local_object'] == k:
                            ids |= set(banks[b['new_bank']].tolist())
                ids = np.array(sorted(ids), dtype=int)
                good = (z['early_2_27_observed'][t, ids] == k) & (z['early_2_27_historical_owner'][t, ids] == k)
                expected[ids[good]] = k
            assert np.array_equal(expected, z[f'comparison_{n}_{mode}'])
            assert int((expected >= 0).sum()) == row[mode]['points']
            assert abs(mass_fraction(n, expected, mapping[t]) - row[mode]['accepted_alpha_coverage']) < 1e-12
    assert not r['cross_gap_identity_assigned'] and not r['PIDG_runtime_modified'] and r['new_optimizer_updates'] == 0
    for name, expected in locks.items():
        assert digest(ROOT / name) == expected, name
    receipt = {'passed': True, 'checked_input_hashes': len(locks), 'domains': checked,
               'bridges_recomputed': len(r['patch_bridges']), 'causal_comparison_frames': 5,
               'complete_native_contribution_integrals': True, 'checkpoint_and_history_unchanged': True,
               'new_optimizer_updates': 0, 'verifier_sha256': digest(Path(__file__))}
    (REPORT / 'verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
