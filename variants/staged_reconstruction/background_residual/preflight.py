"""Native parity/physics backward and independent-process smoke-state audit."""
import argparse
import json
import random
import numpy as np
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..train_static import camera, PIPE
from ..stage2.train import image
from ..stage2.model import compose
from ..union.train import reconstruction as original
from .prepare import EXP, WORK, PARENT, PARENT_SHA, load_parent, verify
from .train import step_loss
from .core import reconstruction, conditional_static_mask, neutralized_target


def records(path):
    return [json.loads(v) for v in path.read_text(encoding='utf-8').splitlines()]


def parity():
    plan = verify(); torch.set_num_threads(4)
    renderer, scene, opt, state, base = load_parent()
    meta = read(DATA/'transforms_train.json'); index = state['camera_stack'][-1]
    frame = dict(meta['frames'][index], _index=index)
    rgba = image(frame, plan['width'])
    gt = torch.tensor(rgba[..., :3]*rgba[..., 3:4], device='cuda').permute(2, 0, 1)
    cam = camera(frame, meta, plan['width'])
    with torch.no_grad():
        a = scene.render(renderer, cam, frame['time'])
        b = renderer(cam, compose(scene.static, [scene.dynamic_at(frame['time'])]), PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
        assert torch.equal(a['render'], b['render']) and torch.equal(a['alpha'], b['alpha'])
        static = renderer(cam, scene.static, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)['render']
    package = scene.render(renderer, cam, frame['time'])
    disabled = reconstruction(package, gt, rgba, scene, cam, enabled=False)
    reference = original(package, gt, rgba, scene, cam)
    assert torch.equal(disabled, reference)
    variables = [scene.pools[0]._xyz, *scene.grids.parameters(), *scene.networks.parameters()]
    g1 = torch.autograd.grad(disabled, variables, retain_graph=True)
    g2 = torch.autograd.grad(reference, variables)
    difference = max(float((a-b).abs().max()) for a, b in zip(g1, g2))
    scale = max(float(g.abs().max()) for g in g2)
    assert difference <= max(1e-7, scale*2e-5), (difference, scale)
    mask = conditional_static_mask(gt)
    target = neutralized_target(gt, static, mask, True)
    assert torch.equal(target[:, ~mask], gt[:, ~mask]) and not target.requires_grad
    del package, disabled, reference, g1, g2
    loss, record = step_loss(scene, renderer, opt, frame, meta, {}, 10, True, plan, base)
    loss.backward()
    assert 'physics' in record
    assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
    assert all(p.grad is None for p in scene.static.parameters())
    gradient = {name: float(sum(p.grad.square().sum() for p in parameters if p.grad is not None).sqrt())
        for name, parameters in [('canonical', [scene.pools[0]._xyz]), ('grid', list(scene.grids.parameters())),
                                 ('deformation', list(scene.networks.parameters())), ('physics', list(scene.physics.parameters()))]}
    assert all(v > 0 for v in gradient.values())
    verify()
    write(EXP/'flags_off.json', {'passed': True, 'native_width': plan['width'],
        'native_RGB_alpha_exact': True, 'original_loss_exact': True, 'gradient_max_abs_difference': difference,
        'gradient_scale': scale, 'CUDA_atomic_tolerance': 'max(1e-7,2e-5*scale)',
        'foreground_empty_unknown_reference_exact': True, 'static_reference_detached': True,
        'actual_original_physics_backward_finite': True, 'candidate_gradient_norms': gradient,
        'optimizer_updates': 0, 'protocol_sha256': sha(EXP/'protocol.json')})
    print('BACKGROUND_RESIDUAL_NATIVE_PREFLIGHT_PASSED', gradient, flush=True)


def audit_arm(arm, expected):
    directory = WORK/'_smoke'/arm
    initial = torch.load(directory/'initial.pt', map_location='cpu', weights_only=False)
    assert sha(directory/'checkpoint.pt') == read(directory/'checkpoint.json')['sha256']
    current = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    for key in parent['model']:
        assert torch.equal(initial['model'][key], parent['model'][key]), key
    assert initial['optimizer']['param_groups'] == parent['optimizer']['param_groups']
    for key, values in parent['optimizer']['state'].items():
        for name, value in values.items():
            assert torch.equal(initial['optimizer']['state'][key][name], value) if torch.is_tensor(value) else initial['optimizer']['state'][key][name] == value
    history = records(directory/'steps.jsonl')
    assert current['iteration'] == expected
    assert [v['local_step'] for v in history] == list(range(1, expected+1))
    rng = random.Random(); rng.setstate(initial['rng_python']); stack = list(initial['camera_stack'])
    for row in history:
        if not stack:
            stack = rng.sample(range(120), 120)
        assert stack.pop() == row['frame_index']
        assert row['neutralization_enabled'] == (arm == 'I10_static_residual')
        assert ('physics' in row) == ((6000+row['local_step']) % 10 == 0)
    assert current['rng_python'] == rng.getstate() and current['camera_stack'] == stack
    assert initial['rng_python'] == parent['rng_python'] and initial['camera_stack'] == parent['camera_stack']
    assert torch.equal(initial['rng_torch'], current['rng_torch']) and torch.equal(initial['rng_torch'], parent['rng_torch'])
    for a, b, c in zip(initial['rng_cuda'], current['rng_cuda'], parent['rng_cuda']):
        assert torch.equal(a, b) and torch.equal(a, c)
    for a, b, c in zip(initial['rng_numpy'], current['rng_numpy'], parent['rng_numpy']):
        assert np.array_equal(a, b) and np.array_equal(a, c)
    physics_ids = {i for group in current['optimizer']['param_groups'] if group['name'] == 'PIDG_velocity_stress' for i in group['params']}
    for i, values in current['optimizer']['state'].items():
        before = initial['optimizer']['state'][i]
        assert float(values['step']-before['step']) == (expected//10 if i in physics_ids else expected)
        assert all(not torch.is_tensor(v) or v.isfinite().all() for v in values.values())
    for key, before in initial['model'].items():
        assert current['model'][key].isfinite().all()
        if key.startswith('static.'):
            assert torch.equal(before, current['model'][key])
    launches = records(directory/'launches.jsonl')
    if arm == 'I10_static_residual':
        assert [v['start'] for v in launches] == [0, 40, 80] and len({v['pid'] for v in launches}) == 3
        for number in [40, 80]:
            value = torch.load(directory/f'step{number:04d}.pt', map_location='cpu', weights_only=False)
            assert value['iteration'] == number
            assert sha(directory/f'step{number:04d}.pt') == launches[[0, 40, 80].index(number)]['checkpoint_sha256']
    return {'arm': arm, 'actual_updates': expected, 'checkpoint_sha256': sha(directory/'checkpoint.pt'),
        'exact_parent_tensors_Adam_RNG_camera_stack': True, 'finite_state_and_exact_Adam_counters': True,
        'sampler_and_all_RNG_replayed': True, 'frozen_static_exact': True,
        'launches': launches, 'physics_updates': expected//10}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--enable-background-residual', action='store_true')
    p.add_argument('mode', choices=['parity', 'audit-smokes']); args = p.parse_args()
    if not args.enable_background_residual:
        raise SystemExit('Background residual preflight is opt-in')
    plan = verify()
    if args.mode == 'parity':
        parity()
    else:
        assert read(EXP/'flags_off.json')['passed']
        assert '8 passed' in (EXP/'unit_tests.log').read_text(encoding='utf-8', errors='replace')
        arms = [audit_arm(arm, plan['smoke_updates']) for arm in plan['arms']]
        schedules = [[v['frame_index'] for v in records(WORK/'_smoke'/arm/'steps.jsonl')] for arm in plan['arms']]
        assert schedules[0] == schedules[1]
        verify()
        write(EXP/'preflight.json', {'passed': True, 'arms': arms, 'unit_tests': 8,
            'flags_off_sha256': sha(EXP/'flags_off.json'), 'source_freeze_sha256': sha(EXP/'source_freeze.json'),
            'matched_camera_schedule': True, 'candidate_independent_process_restart_40_80': True,
            'PIDG_baseline_sources_unchanged': True, 'new_physics_or_material_modules': False})
        print('BACKGROUND_RESIDUAL_SMOKE_AUDIT_PASSED', flush=True)


if __name__ == '__main__':
    main()
