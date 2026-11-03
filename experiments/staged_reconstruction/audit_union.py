"""Independent final count-free state, sampler, objective and lineage audit."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from variants.staged_reconstruction.prepare_data import REPORT, RUNS, DATA, read, write, sha
from variants.staged_reconstruction.union.prepare import UNION_REPORT, UNION_RUNS, verify_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['warmup', 'physics'], required=True)
    args = parser.parse_args()
    verify_sources()
    phase = args.phase
    directory = UNION_RUNS/phase/'formal'
    checkpoint = directory/'checkpoint.pt'
    before = sha(checkpoint)
    assert before == read(directory/'checkpoint.json')['sha256']
    protocol = read(UNION_REPORT/'protocol.json')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert sha(directory/'initial.pt') == read(directory/'initial.json')['sha256']
    initial = torch.load(directory/'initial.pt', map_location='cpu', weights_only=False)
    assert state['phase'] == phase and state['iteration'] == protocol[phase+'_updates']
    assert state['protocol_sha256'] == initial['protocol_sha256'] == sha(UNION_REPORT/'protocol.json')
    assert state['physical_body_number'] is None and not state['instance_identity_available']
    static = torch.load(RUNS/'S1_static/checkpoint.pt', map_location='cpu', weights_only=False)['model']
    assert all(torch.equal(v, state['model']['static.'+k]) for k, v in static.items())
    assert all(v.isfinite().all() for v in state['model'].values())
    assert all(v.isfinite().all() for s in state['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v))
    assert len(state['physics_graphs']) == 1
    assert not any('object_ids' in k for k in state['model'])
    assert state['model']['pools.0._xyz'].shape == (protocol['dynamic_point_budget'], 3)
    assert not any(k.startswith('pools.1.') or k.startswith('grids.1.') or k.startswith('physics.fields.1.') for k in state['model'])
    rng = random.Random()
    rng.setstate(initial['rng_python'])
    stack = list(initial['camera_stack'])
    sequence = []
    for _ in range(state['iteration']):
        if not stack:
            stack = rng.sample(range(120), 120)
        sequence.append(stack.pop())
    assert rng.getstate() == state['rng_python'] and stack == state['camera_stack']
    actual = [json.loads(line) for line in (directory/'sampling.jsonl').read_text().splitlines()]
    assert [r['step'] for r in actual] == list(range(1, state['iteration']+1))
    assert [r['frame_index'] for r in actual] == sequence
    counters = {str(k): float(v['step']) for k, v in state['optimizer']['state'].items()}
    physics_ids = {str(k) for g in state['optimizer']['param_groups'] if g['name'] == 'PIDG_velocity_stress' for k in g['params']}
    for key, count in counters.items():
        assert count == (600 if key in physics_ids else (16000 if phase == 'warmup' else 22000)), (key, count)
    if phase == 'warmup':
        assert not physics_ids.intersection(counters)
        assert state['parent_sha256'] == protocol['static_checkpoint_sha256']
    else:
        assert state['parent_sha256'] == sha(UNION_RUNS/'warmup/formal/checkpoint.pt')
        lines = [json.loads(line) for line in (directory/'physics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in lines] == list(range(10, state['iteration']+1, 10))
        assert all(not r['physical_boundary_exclusion_available'] for r in lines)
        for row in lines:
            assert len(row['population_diagnostics']) == 1
            assert row['population_diagnostics'][0]['valid_fraction'] >= .25
    val = read(UNION_REPORT/phase/f'val_{state["iteration"]:05d}_1600.json')['metrics']
    original_static = read(REPORT/'S1_final_verification.json')['validation']['candidate_static_PSNR']
    gates = {'dynamic_PSNR': val['dynamic_PSNR'] >= 20., 'dynamic_only_alpha': val['dynamic_alpha'] >= .85,
        'static_preservation': original_static-val['static_PSNR'] <= 1.}
    assert sha(checkpoint) == before
    result = {'phase': phase, 'checkpoint_sha256': before, 'iteration': state['iteration'], 'parent_sha256': state['parent_sha256'],
        'all_frozen_sources_inputs_protocols_match': True, 'static_exact_S1': True, 'finite_model_and_optimizer': True,
        'one_dynamic_population_and_shared_field': True, 'physical_body_number': None, 'instance_identity_available': False,
        'exact_all_step_sampling_and_final_python_RNG': True, 'sampler_sequence_sha256': hashlib.sha256(json.dumps(sequence).encode()).hexdigest(),
        'optimizer_state_count': len(counters), 'all_actual_Adam_counters_correct': True, 'native_validation': val,
        'numeric_gates': gates, 'parameters': read(directory/'initial.json'),
        'physical_boundary_exclusion_available': False, 'manual_image_review_pending': True}
    if phase == 'physics':
        result['native_test'] = read(UNION_REPORT/phase/'test_06000_1600.json')['metrics']
        result['physics_objective_calls'] = len(lines)
        result['physics_valid_fraction_min'] = min(row['population_diagnostics'][0]['valid_fraction'] for row in lines)
    write(UNION_REPORT/f'{phase}_state_audit.json', result)
    print('UNION_STATE_AUDIT', phase, gates, flush=True)


if __name__ == '__main__':
    main()
