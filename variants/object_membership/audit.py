"""Training-free object association from immutable native-resolution C1 caches."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from output.surface_support.core import propagate
from output.surface_correspondence.audit import center_inside
from variants.flow_comparison.geometry import sample, forward_backward_valid
from variants.surface_correspondence.model import qualify_fragments, particle_support
from variants.object_membership.model import Config, optional_membership, reciprocal_mapping

REPORT = ROOT / 'experiments/object_membership'
OUT = ROOT / 'output/object_membership'
CACHE = ROOT / 'runs/pidg_short_clip_diagnostic/cache'
FLOW = ROOT / 'data/derived/flow_comparison_train_only'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def arrays(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def image_links(frames, components, pairs):
    """Each prefix is checked in both directions; reverse check uses no later RGB."""
    labels = components[frames[0]].copy()
    alive = np.ones(4, bool)
    rows = []
    mappings = []
    for t, number in enumerate(frames):
        if t:
            labels, _ = propagate(labels, components[number], *pairs[frames[t - 1], number])
        forward, alive = qualify_fragments(labels, components[number], alive)
        for seed in range(4):
            if not alive[seed]:
                labels[labels == seed] = -1
        back_labels = components[number].copy()
        back_alive = np.ones(4, bool)
        backward = np.arange(4, dtype=np.int16)
        for j in range(t, 0, -1):
            f, b = pairs[frames[j - 1], frames[j]]
            back_labels, _ = propagate(back_labels, components[frames[j - 1]], b, f)
            backward, back_alive = qualify_fragments(back_labels, components[frames[j - 1]], back_alive)
            for seed in range(4):
                if not back_alive[seed]:
                    back_labels[back_labels == seed] = -1
        verified = reciprocal_mapping(forward, backward)
        mappings.append(verified)
        rows.append({'frame': number, 'forward_seed_to_component': forward.tolist(),
                     'reverse_component_to_seed': backward.tolist(), 'reciprocal_seed_to_component': verified.tolist()})
    return np.asarray(mappings), rows


def motion_error(source, target, pair):
    f, b = pair
    p = torch.as_tensor(source['means'])
    q = torch.as_tensor(target['means'])
    valid = sample(forward_backward_valid(f, b).float(), p[:, None])[:, 0] > .99
    valid &= (p[:, 0] >= 0) & (p[:, 0] <= 399) & (p[:, 1] >= 0) & (p[:, 1] <= 224)
    error = torch.linalg.vector_norm(q - p - sample(f, p[:, None])[:, 0], dim=-1).numpy()
    error[~valid.numpy()] = np.inf
    return error


def point_labels(frames, mapping, components, geometry, native, pairs):
    result = []
    previous_support = None
    for t, number in enumerate(frames):
        value = geometry[number]
        support = np.zeros((4, len(value['means'])), bool)
        for seed, component in enumerate(mapping[t]):
            if component < 0:
                continue
            inside = center_inside(value['means'], value['depth'], value['cov'], components[number] == component)
            support[seed] = particle_support(native[number]['mass'][:, component], native[number]['mass'][:, 5], inside)
        joint = support.copy()
        if t:
            error = motion_error(geometry[frames[t - 1]], value, pairs[frames[t - 1], number])
            joint &= previous_support & (np.isfinite(error) & (error <= 2))[None]
        labels = np.full(support.shape[1], -1, np.int16)
        unique = joint.sum(axis=0) == 1
        labels[unique] = joint[:, unique].argmax(axis=0)
        result.append(labels)
        previous_support = support
    return np.asarray(result)


def coverage(number, labels, mapping, components, native):
    mass = 0.
    for seed, component in enumerate(mapping):
        if component >= 0:
            mass += float(native[number]['mass'][labels == seed, component].sum(dtype=np.float64))
    denominator = float(native[number]['alpha'][components[number] >= 0].sum(dtype=np.float64))
    return mass / denominator


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    plan = read(REPORT / 'plan.json')
    old = read(ROOT / 'experiments/short_clip_diagnostic/resolution_control.json')
    model = read(ROOT / 'experiments/short_clip_diagnostic/model_results.json')
    manifest = read(FLOW / 'manifest.json')
    track = arrays(ROOT / 'output/surface_support/image_tracks.npz')
    components = {int(n): c for n, c in zip(track['frames'], track['components'])}
    numbers = sorted({n for frames in plan['domains'].values() for n in frames})
    keys = {(a, b) for frames in plan['domains'].values() for a, b in zip(frames[:-1], frames[1:])}
    flow_rows = {(r['source'], r['target']): r for r in manifest['pairs']}
    expected = read(ROOT / 'experiments/short_clip_diagnostic/model_input_lock.json')
    expected.update(read(ROOT / 'experiments/short_clip_diagnostic/cache_manifest.json'))
    for bank in old['native_seed_banks']:
        if bank['qualified']:
            expected[bank['path']] = bank['sha256']
    checkpoint = 'runs/pidg_color_repair/C1_repaired/rubber_duck/checkpoints/resume.pt'
    expected[checkpoint] = model['checkpoint_sha256']
    for key in keys:
        file = flow_rows[key]['file']
        expected[str((FLOW / file).relative_to(ROOT))] = manifest['output_sha256'][file]
    for path, digest in expected.items():
        assert sha(ROOT / path) == digest, path
    own = [Path(__file__), REPORT / 'plan.json', ROOT / 'variants/object_membership/model.py',
           ROOT / 'tests/test_object_membership.py', ROOT / 'experiments/short_clip_diagnostic/resolution_control.json',
           ROOT / 'experiments/short_clip_diagnostic/model_results.json', ROOT / 'output/surface_support/image_tracks.npz']
    expected.update({str(p.relative_to(ROOT)): sha(p) for p in own})
    write(REPORT / 'input_lock.json', expected)
    geometry = {n: {k: v for k, v in arrays(CACHE / f'frame{n:04d}.npz').items() if k in ['means', 'depth', 'cov']} for n in numbers}
    native = {n: arrays(CACHE / f'native{n:04d}.npz') for n in numbers}
    pairs = {}
    for key in keys:
        z = arrays(FLOW / flow_rows[key]['file'])
        pairs[key] = (torch.tensor(z['forward'], dtype=torch.float32), torch.tensor(z['backward'], dtype=torch.float32))
    saved = {'checkpoint_sha256': np.asarray(model['checkpoint_sha256']), 'protocol_version': np.asarray(1)}
    domains = []
    for domain, frames in plan['domains'].items():
        mapping, links = image_links(frames, components, pairs)
        observed = point_labels(frames, mapping, components, geometry, native, pairs)
        ledger = optional_membership(observed, frames, Config(enabled=True))
        saved[domain + '_frames'] = np.asarray(frames)
        saved[domain + '_mapping'] = mapping
        saved[domain + '_observed'] = observed
        for key, value in ledger.items():
            saved[domain + '_' + key] = value
        rows = []
        for t, n in enumerate(frames):
            owner = ledger['historical_owner'][t]
            visible = ledger['visible_owner'][t]
            rows.append({'frame': n, 'supported_points': int((observed[t] >= 0).sum()),
                         'historical_member_points': int((owner >= 0).sum()), 'currently_visible_member_points': int((visible >= 0).sum()),
                         'latched_conflict_points': int(ledger['conflict'][t].sum()),
                         'historical_member_points_per_object': [int((owner == k).sum()) for k in range(4)],
                         'current_visible_member_alpha_coverage': coverage(n, visible, mapping[t], components, native),
                         'instantaneous_supported_alpha_coverage': coverage(n, observed[t], mapping[t], components, native)})
        domains.append({'domain': domain, 'links': links, 'frames': rows,
                        'final_conflict_row_ids': np.flatnonzero(ledger['conflict'][-1]).tolist(),
                        'unique_ever_supported_points': int((observed >= 0).any(axis=0).sum()),
                        'ownership_scope': 'conditional local image object, not material identity or global re-identification'})
        print(domain, rows[-1], flush=True)
    # Connect A15 bank seed labels to the continuously tracked A2 domain.
    frames = plan['domains']['early_2_27']
    mapping = saved['early_2_27_mapping']
    observed = saved['early_2_27_observed']
    banks = {b['name']: arrays(ROOT / b['path'])['kernel_ids'] for b in old['native_seed_banks'] if b['qualified']}
    bridge_rows = []
    approved = {}
    for seed in range(4):
        name = f'A15_F{seed}'
        candidate = []
        for n in [15, 16, 19]:
            region = next(row['region'] for row in model['bank_frames'] if row['bank'] == name and row['frame'] == n)
            candidates = np.flatnonzero(mapping[frames.index(n)] == region)
            candidate.append(int(candidates[0]) if len(candidates) == 1 else -1)
        object_id = candidate[0] if min(candidate) >= 0 and len(set(candidate)) == 1 else -1
        approved[seed] = object_id
        current = banks[name]
        intersection = [int(np.intersect1d(current, banks[f'A2_F{k}']).size) for k in range(4)]
        bridge_rows.append({'new_bank': name, 'seed_frames': [15, 16, 19], 'early_local_object': object_id,
                            'verified_object_at_each_seed': candidate, 'particle_count': len(current),
                            'row_overlap_with_A2_objects': intersection,
                            'rows_new_to_same_object_A2_bank': int(np.setdiff1d(current, banks[f'A2_F{object_id}']).size) if object_id >= 0 else None,
                            'same_material_surface_claimed': False})
    comparison = []
    for t, n in enumerate(frames):
        if n <= 19:
            continue
        modes = {}
        for mode in ['A2_only', 'A2_plus_A15']:
            selected = np.full(observed.shape[1], -1, np.int16)
            for seed in range(4):
                ids = banks[f'A2_F{seed}']
                if mode == 'A2_plus_A15':
                    ids = np.union1d(ids, np.concatenate([banks[f'A15_F{k}'] for k in range(4) if approved[k] == seed])) if seed in approved.values() else ids
                # Causal ledger: future contradictions cannot suppress past outputs.
                ok = (observed[t, ids] == seed) & (saved['early_2_27_historical_owner'][t, ids] == seed)
                selected[ids[ok]] = seed
            saved[f'comparison_{n}_{mode}'] = selected
            modes[mode] = {'points': int((selected >= 0).sum()), 'accepted_alpha_coverage': coverage(n, selected, mapping[t], components, native)}
        comparison.append({'frame': n, **modes})
    np.savez_compressed(OUT / 'membership.npz', **saved)
    result = {'protocol_version': 1, 'checkpoint_sha256': model['checkpoint_sha256'], 'Gaussian_rows': observed.shape[1],
              'render_width': 1600, 'mask_and_center_flow_width': 400, 'domains': domains, 'patch_bridges': bridge_rows,
              'causal_bank_comparison_after_A15_seeds': comparison, 'cross_gap_identity_assigned': False,
              'new_optimizer_updates': 0, 'PIDG_runtime_modified': False, 'trained_material_segmentation': False,
              'arrays_sha256': sha(OUT / 'membership.npz'), 'input_file_count': len(expected)}
    write(REPORT / 'results.json', result)
    assert all(sha(ROOT / p) == v for p, v in expected.items())
    print('OBJECT MEMBERSHIP AUDIT COMPLETE; model and protected inputs unchanged', flush=True)


if __name__ == '__main__':
    main()
