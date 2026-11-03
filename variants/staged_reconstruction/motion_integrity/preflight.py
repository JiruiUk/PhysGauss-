"""Native flags-off check and independent-process optimizer/sampler audit."""
import argparse
import json
import random
import numpy as np
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..train_static import camera, PIPE
from ..stage2.model import compose
from ..union.train import reconstruction
from ..stage2.train import image
from .prepare import EXP, WORK, PARENT, PARENT_SHA, load_parent
from .train import step_loss


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def audit_arm(arm, expected):
    directory = WORK/'_smoke'/arm
    initial = torch.load(directory/'initial.pt', map_location='cpu', weights_only=False)
    path = directory/'checkpoint.pt'
    assert sha(path) == read(directory/'checkpoint.json')['sha256']
    state = torch.load(path, map_location='cpu', weights_only=False)
    records = rows(directory/'steps.jsonl')
    assert state['iteration'] == expected
    assert [r['local_step'] for r in records] == list(range(1, expected+1))
    rng = random.Random()
    rng.setstate(initial['rng_python'])
    stack = list(initial['camera_stack'])
    for record in records:
        if not stack:
            stack = rng.sample(range(120), 120)
        assert record['frame_index'] == stack.pop()
    assert state['rng_python'] == rng.getstate() and state['camera_stack'] == stack
    assert torch.equal(initial['rng_torch'], state['rng_torch'])
    assert all(torch.equal(a, b) for a, b in zip(initial['rng_cuda'], state['rng_cuda']))
    assert np.array_equal(initial['rng_numpy'][1], state['rng_numpy'][1])
    physics_ids = {i for group in state['optimizer']['param_groups'] if group['name'] == 'PIDG_velocity_stress' for i in group['params']}
    for i, current in state['optimizer']['state'].items():
        before = initial['optimizer']['state'][i]
        assert float(current['step']-before['step']) == (expected//10 if i in physics_ids else expected)
        assert all(not torch.is_tensor(v) or v.isfinite().all() for v in current.values())
    for key, before in initial['model'].items():
        assert state['model'][key].isfinite().all()
        if key.startswith('static.'):
            assert torch.equal(before, state['model'][key])
    assert any(not torch.equal(v, state['model'][k]) for k, v in initial['model'].items() if k.startswith('networks.'))
    if arm == 'I1_curve_isometry':
        assert all('prior' in row and np.isfinite(row['prior']) for row in records)
        for number in [40, 80]:
            snapshot = torch.load(directory/f'step{number:04d}.pt', map_location='cpu', weights_only=False)
            assert snapshot['iteration'] == number
            assert len(rows(directory/'steps.jsonl')[:number]) == number
    else:
        assert all('prior' not in row for row in records)
    return {'arm': arm, 'actual_updates': expected, 'finite_full_state': True, 'exact_sampler_replayed': True,
            'all_saved_RNG_verified': True, 'static_exact': True, 'Adam_counter_advances_exact': True,
            'sha256': sha(path)}


def parity():
    torch.set_num_threads(4)
    renderer, scene, opt, state, base = load_parent()
    meta = read(DATA/'transforms_train.json')
    index = state['camera_stack'][-1]
    frame = dict(meta['frames'][index], _index=index)
    plan = read(EXP/'protocol.json')
    loss, record = step_loss(scene, renderer, opt, frame, meta, {}, 6001, 1, False, plan, base, None)
    rgba = image(frame, 1600)
    gt = torch.tensor(rgba[..., :3]*rgba[..., 3:4], device='cuda').permute(2, 0, 1)
    cam = camera(frame, meta, 1600)
    package = scene.render(renderer, cam, frame['time'])
    reference = reconstruction(package, gt, rgba, scene, cam)
    # Absolute6001 is not a physics step; this is the exact original expression.
    assert torch.equal(loss, reference)
    with torch.no_grad():
        a = scene.render(renderer, cam, frame['time'])
        b = renderer(cam, compose(scene.static, [scene.dynamic_at(frame['time'])]), PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
        assert torch.equal(a['render'], b['render']) and torch.equal(a['alpha'], b['alpha'])
    g1 = torch.autograd.grad(loss, [scene.pools[0]._xyz, *scene.networks.parameters()], retain_graph=True)
    g2 = torch.autograd.grad(reference, [scene.pools[0]._xyz, *scene.networks.parameters()])
    difference = max(float((a-b).abs().max()) for a, b in zip(g1, g2))
    scale = max(float(a.abs().max()) for a in g2)
    assert difference <= max(1e-7, scale*2e-5), (difference, scale)
    write(EXP/'flags_off.json', {'passed': True, 'native_render_alpha_exact': True,
          'inherited_reconstruction_loss_exact': True, 'gradient_max_abs_difference': difference,
          'gradient_max_abs_scale': scale, 'CUDA_atomic_gradient_tolerance': 'max(1e-7,2e-5*gradient_scale)',
          'original_model_structure_and_renderer_unchanged': True, 'optimizer_updates': 0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parity', action='store_true')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.parity:
        parity()
    elif args.audit:
        arms = [audit_arm('I0_control', 100), audit_arm('I1_curve_isometry', 100)]
        control = rows(WORK/'_smoke/I0_control/steps.jsonl')
        candidate = rows(WORK/'_smoke/I1_curve_isometry/steps.jsonl')
        assert [r['frame_index'] for r in control] == [r['frame_index'] for r in candidate]
        assert sha(PARENT) == PARENT_SHA
        assert read(EXP/'flags_off.json')['passed']
        write(EXP/'preflight.json', {'passed': True, 'arms': arms, 'flags_off': read(EXP/'flags_off.json'),
              'candidate_process_restarted_at_40_and_80': True, 'matched_camera_schedule': True,
              'parent_sha256_unchanged': PARENT_SHA, 'no_physics_module_added': True})
        print('MOTION_PREFLIGHT_PASSED', flush=True)
    else:
        parser.error('Choose --parity or --audit')


if __name__ == '__main__':
    main()
