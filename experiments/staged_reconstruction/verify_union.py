"""Read-only real-state recovery, static freeze and original-renderer checks."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from variants.staged_reconstruction.prepare_data import REPORT, RUNS, DATA, read, write, sha
from variants.staged_reconstruction.union.prepare import UNION_REPORT, UNION_RUNS, verify_sources


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def snapshot(state):
    return {'iteration': state['iteration'], 'model': {k: digest(v) for k, v in state['model'].items()},
        'counters': {str(k): float(v['step']) for k, v in state['optimizer']['state'].items()},
        'physics_ids': [str(k) for group in state['optimizer']['param_groups'] if group['name'] == 'PIDG_velocity_stress' for k in group['params']],
        'camera_stack': state['camera_stack'], 'rng_python': state['rng_python']}


def verify_counter_advances(prior, state, phase):
    current = snapshot(state)
    assert set(current['counters']) == set(prior['counters'])
    for key, value in current['counters'].items():
        assert value-prior['counters'][key] == (4 if key in current['physics_ids'] else 40), key
    rng = random.Random()
    old = prior['rng_python']
    rng.setstate((old[0], tuple(old[1]), old[2]))
    stack = list(prior['camera_stack'])
    for _ in range(40):
        if not stack:
            stack = rng.sample(range(120), 120)
        stack.pop()
    assert rng.getstate() == state['rng_python'] and stack == state['camera_stack']
    changed = [k for k, v in current['model'].items() if v != prior['model'][k]]
    assert any(k.startswith('networks.') for k in changed)
    if phase == 'physics':
        assert any(k.startswith('physics.') for k in changed)
    else:
        assert not any(k.startswith('physics.') for k in changed)
    return len(current['counters']), len(changed)


def native_preflight():
    from variants.staged_reconstruction.train_static import setup, camera, PIPE
    renderer = setup()
    from variants.staged_reconstruction.union.model import UnionScene
    from variants.staged_reconstruction.stage2.model import compose
    from scene_PIDG.gaussian_model import GaussianModel
    protocol = read(UNION_REPORT/'protocol.json')
    scene = UnionScene(protocol)
    meta = read(DATA/'transforms_train.json')
    cam = camera(meta['frames'][0], meta, 800)
    with torch.no_grad():
        combined = compose(scene.static, [scene.dynamic_at(0.)])
        original = GaussianModel(3)
        original.active_sh_degree = 3
        original._xyz = combined.get_xyz
        original._scaling = combined.get_scaling.log()
        original._rotation = combined.get_rotation
        original._opacity = torch.logit(combined.get_opacity)
        original._features_dc = combined.get_features[:, :1]
        original._features_rest = combined.get_features[:, 1:]
        a = renderer(cam, combined, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
        b = renderer(cam, original, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
        # Activation round trips are FP32, so measure the tiny numeric difference.
        delta = float((a['render']-b['render']).abs().max())
        assert delta < 2e-6, delta
        assert len(scene.pools) == len(scene.deforms) == len(scene.physics.fields) == 1
        assert not any(key.endswith('object_ids') for key in scene.state_dict())
    write(UNION_REPORT/'native_preflight.json', {'passed': True, 'original_GaussianModel_renderer_max_abs_difference': delta,
        'single_dynamic_population': True, 'single_original_PIDG_deformation_and_velocity_stress': True,
        'physical_body_number': None, 'static_source_sha256': sha(RUNS/'S1_static/checkpoint.pt')})
    print('UNION_NATIVE_PREFLIGHT', delta, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['warmup', 'physics'], default='warmup')
    parser.add_argument('--capture40', action='store_true')
    parser.add_argument('--native-preflight', action='store_true')
    args = parser.parse_args()
    verify_sources()
    if args.native_preflight:
        native_preflight()
        return
    directory = UNION_RUNS/args.phase/'_smoke'
    path = directory/'checkpoint.pt'
    assert sha(path) == read(directory/'checkpoint.json')['sha256']
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['phase'] == args.phase and state['protocol_sha256'] == sha(UNION_REPORT/'protocol.json')
    assert state['physical_body_number'] is None and not state['instance_identity_available']
    if args.capture40:
        assert state['iteration'] == 40
        write(UNION_REPORT/f'{args.phase}_smoke40.json', snapshot(state))
        print('UNION_SMOKE40_CAPTURED', args.phase, flush=True)
        return
    assert state['iteration'] == 80
    counters, changed = verify_counter_advances(read(UNION_REPORT/f'{args.phase}_smoke40.json'), state, args.phase)
    static = torch.load(RUNS/'S1_static/checkpoint.pt', map_location='cpu', weights_only=False)['model']
    assert all(torch.equal(value, state['model']['static.'+key]) for key, value in static.items())
    assert all(v.isfinite().all() for v in state['model'].values())
    assert all(v.isfinite().all() for s in state['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v))
    assert len(state['physics_graphs']) == 1
    if args.phase == 'physics':
        lines = [json.loads(line) for line in (directory/'physics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in lines] == list(range(10, 81, 10))
        assert all(not r['physical_boundary_exclusion_available'] for r in lines)
    result = {'passed': True, 'phase': args.phase, 'independent_process_resume_40_to_80': True,
        'checkpoint_sha256': sha(path), 'finite_model_and_optimizer': True,
        'optimizer_state_count': counters, 'changed_parameter_tensors': changed,
        'exact_camera_and_python_RNG_replay': True, 'all_RNG_payloads_present': all(k in state for k in ['rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda']),
        'static_exact_S1': True, 'derived_graph_state_saved': True, 'physical_body_number': None}
    write(UNION_REPORT/f'{args.phase}_smoke_verification.json', result)
    print('UNION_SMOKE_VERIFIED', args.phase, counters, flush=True)


if __name__ == '__main__':
    main()
