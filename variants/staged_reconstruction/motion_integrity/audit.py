"""Independent graph, inherited state and matched-training receipts."""
import argparse
import json
import random
import numpy as np
import torch
from ..prepare_data import ROOT, REPORT, read, write, sha
from ..union.prepare import verify_sources
from .prepare import EXP, WORK, PARENT, PARENT_SHA


def files():
    return [*sorted((ROOT/'variants/staged_reconstruction/motion_integrity').glob('*.py')),
            ROOT/'tests/test_motion_integrity.py', EXP/'protocol.json', EXP/'research.md',
            WORK/'graph.npz', PARENT]


def verify_graph():
    with np.load(WORK/'graph.npz') as g:
        edges, positions = g['edges'], g['parent_positions']
        assert (edges[:, 0] < edges[:, 1]).all()
        assert len(np.unique(edges, axis=0)) == len(edges)
        assert np.bincount(edges.ravel()).max() <= 12
        error = 0.
        for start in range(0, len(edges), 2048):
            e = edges[start:start+2048]
            # Separate scalar-coordinate formula, not the graph constructor.
            delta = positions[:, e[:, 1]].astype(np.float64)-positions[:, e[:, 0]].astype(np.float64)
            distances = np.sqrt(delta[..., 0]**2+delta[..., 1]**2+delta[..., 2]**2)
            assert distances.max() <= .080001
            actual_rest = np.median(distances, axis=0)
            error = max(error, float(np.max(np.abs(actual_rest-g['rest'][start:start+2048]))))
            assert np.allclose(distances.max(0), g['curve_max'][start:start+2048], atol=1e-7)
            assert np.allclose(np.exp(-.5*(distances.max(0)/.08)**2), g['weight'][start:start+2048], atol=2e-7)
        assert error < 1e-7
        return {'edges': len(edges), 'degree_max': 12, 'independent_rest_length_max_abs_error': error,
                'train_source_time_count': len(g['source_indices'])}


def audit_training():
    plan = read(EXP/'protocol.json')
    results, sequences = {}, []
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    for arm in plan['arms']:
        folder = WORK/'formal'/arm
        path = folder/'checkpoint.pt'
        assert sha(path) == read(folder/'checkpoint.json')['sha256']
        state = torch.load(path, map_location='cpu', weights_only=False)
        initial = torch.load(folder/'initial.pt', map_location='cpu', weights_only=False)
        assert state['iteration'] == plan['actual_updates']
        assert state['parent_sha256'] == PARENT_SHA
        assert state['protocol_sha256'] == sha(EXP/'protocol.json')
        assert all(torch.equal(v, parent['model'][k]) for k, v in initial['model'].items())
        assert initial['camera_stack'] == parent['camera_stack'] and initial['rng_python'] == parent['rng_python']
        for idx, value in parent['optimizer']['state'].items():
            assert all(torch.equal(v, initial['optimizer']['state'][idx][k]) if torch.is_tensor(v) else v == initial['optimizer']['state'][idx][k] for k, v in value.items())
        records = [json.loads(v) for v in (folder/'steps.jsonl').read_text().splitlines()]
        assert [r['local_step'] for r in records] == list(range(1, plan['actual_updates']+1))
        sequence = [r['frame_index'] for r in records]
        sequences.append(sequence)
        rng, stack = random.Random(), list(parent['camera_stack'])
        rng.setstate(parent['rng_python'])
        for index in sequence:
            if not stack:
                stack = rng.sample(range(120), 120)
            assert index == stack.pop()
        assert stack == state['camera_stack'] and state['rng_python'] == rng.getstate()
        assert torch.equal(state['rng_torch'], parent['rng_torch'])
        assert all(torch.equal(a, b) for a, b in zip(parent['rng_cuda'], state['rng_cuda']))
        physics_ids = {i for group in state['optimizer']['param_groups'] if group['name'] == 'PIDG_velocity_stress' for i in group['params']}
        assert len([r for r in records if 'physics' in r]) == plan['actual_updates']//10
        for idx, value in state['optimizer']['state'].items():
            assert float(value['step']-parent['optimizer']['state'][idx]['step']) == (plan['actual_updates']//10 if idx in physics_ids else plan['actual_updates'])
            assert all(not torch.is_tensor(v) or v.isfinite().all() for v in value.values())
        assert all(v.isfinite().all() for v in state['model'].values())
        assert all(torch.equal(v, parent['model'][k]) for k, v in state['model'].items() if k.startswith('static.'))
        results[arm] = {'actual_optimizer_updates': state['iteration'], 'sha256': sha(path),
                        'physics_updates': plan['actual_updates']//10, 'initial_full_state_matches_parent': True,
                        'finite_model_and_optimizer': True, 'camera_RNG_and_Adam_replayed': True, 'static_frozen_exact': True,
                        'initial_100_prior_mean': float(np.mean([r['prior'] for r in records[:100]])) if 'prior' in records[0] else None,
                        'final_100_prior_mean': float(np.mean([r['prior'] for r in records[-100:]])) if 'prior' in records[0] else None}
    assert sequences[0] == sequences[1]
    return {'arms': results, 'all_camera_samples_matched': True, 'test_RGB_used_for_selection': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--formal', action='store_true')
    args = p.parse_args()
    verify_sources()
    assert sha(PARENT) == PARENT_SHA
    graph = verify_graph()
    if args.freeze:
        assert read(EXP/'preflight.json')['passed']
        assert not (EXP/'source_freeze.json').exists()
        write(EXP/'source_freeze.json', {f.relative_to(ROOT).as_posix(): sha(f) for f in files()})
        write(EXP/'graph_verification.json', graph)
    elif args.formal:
        for name, expected in read(EXP/'source_freeze.json').items():
            assert sha(ROOT/name) == expected, name
        result = audit_training()
        write(EXP/'training_verification.json', {'passed': True, 'source_freeze_unchanged': True,
              'graph_independently_checked': graph, **result})
    else:
        p.error('Choose --freeze or --formal')
    print('MOTION_AUDIT_PASSED', 'freeze' if args.freeze else 'formal', flush=True)


if __name__ == '__main__':
    main()
