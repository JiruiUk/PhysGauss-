"""Real-parent shared mapping checks and actual full-state recovery audit."""
import argparse
import json
import random
import numpy as np
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..train_static import camera
from ..motion_integrity.prepare import load_parent, PARENT, PARENT_SHA
from .prepare import EXP, WORK
from .train import install


def native():
    torch.set_num_threads(4)
    renderer, scene, _, _, _ = load_parent()
    meta = read(DATA/'transforms_train.json')
    cam = camera(meta['frames'][0], meta, 1600)
    with torch.no_grad():
        original = scene.render(renderer, cam, .43)
    parameters = sum(p.numel() for p in scene.parameters())
    adapter = install(scene, read(EXP/'protocol.json'))
    with torch.no_grad():
        disabled = scene.render(renderer, cam, .43)
        assert torch.equal(original['render'], disabled['render'])
        assert torch.equal(original['alpha'], disabled['alpha'])
        errors = {}
        from variants.flow_comparison.temporal import position
        for mix in [0., .25, 1.]:
            adapter.mix = mix
            direct = scene.dynamic_at(.43).get_xyz
            queried = position(adapter, scene.pools[0].get_xyz, .43)+scene.centers[0]
            error = float((direct-queried).abs().max())
            assert error < 3e-5, (mix, error)
            errors[str(mix)] = error
    adapter.mix = 1.
    x = scene.pools[0].get_xyz
    mapped = scene.dynamic_at(.43).get_xyz
    loss = mapped.square().mean()
    gradients = torch.autograd.grad(loss, [x, *scene.grids.parameters(), *scene.networks.parameters()], allow_unused=True)
    assert all(g is None or g.isfinite().all() for g in gradients)
    assert gradients[0].abs().sum() > 0
    assert sum(float(g.abs().sum()) for g in gradients[1:] if g is not None) > 0
    assert sum(p.numel() for p in scene.parameters()) == parameters
    write(EXP/'native_preflight.json', {'passed': True, 'flags_off_native_rgb_alpha_exact': True,
          'render_and_physics_full_center_max_abs_difference': errors,
          'full_sparse_intended_gradients_finite_and_nonzero': True, 'no_new_trainable_parameters': True,
          'optimizer_updates': 0, 'parent_unchanged': sha(PARENT) == PARENT_SHA})
    print('SPARSE_NATIVE_PREFLIGHT', errors, flush=True)


def audit(smoke):
    directory = WORK/('_smoke' if smoke else 'formal')
    expected = 500 if smoke else read(EXP/'protocol.json')['actual_updates']
    path = directory/'checkpoint.pt'
    assert sha(path) == read(directory/'checkpoint.json')['sha256']
    state = torch.load(path, map_location='cpu', weights_only=False)
    initial = torch.load(directory/'initial.pt', map_location='cpu', weights_only=False)
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    assert all(torch.equal(v, initial['model'][k]) for k, v in parent['model'].items())
    for idx, value in parent['optimizer']['state'].items():
        assert all(torch.equal(v, initial['optimizer']['state'][idx][k]) if torch.is_tensor(v) else v == initial['optimizer']['state'][idx][k] for k, v in value.items())
    assert state['iteration'] == expected and state['protocol_sha256'] == sha(EXP/'protocol.json')
    rows = [json.loads(v) for v in (directory/'steps.jsonl').read_text().splitlines()]
    assert [r['local_step'] for r in rows] == list(range(1, expected+1))
    rng, stack = random.Random(), list(parent['camera_stack'])
    rng.setstate(parent['rng_python'])
    for row in rows:
        if not stack:
            stack = rng.sample(range(120), 120)
        assert row['frame_index'] == stack.pop()
        assert row['sparse_mix'] == min(1., row['local_step']/400)
    assert state['rng_python'] == rng.getstate() and state['camera_stack'] == stack
    assert torch.equal(parent['rng_torch'], state['rng_torch'])
    assert all(torch.equal(a, b) for a, b in zip(parent['rng_cuda'], state['rng_cuda']))
    assert np.array_equal(parent['rng_numpy'][1], state['rng_numpy'][1])
    ids = {idx for group in state['optimizer']['param_groups'] if group['name'] == 'PIDG_velocity_stress' for idx in group['params']}
    for idx, value in state['optimizer']['state'].items():
        assert float(value['step']-parent['optimizer']['state'][idx]['step']) == (expected//10 if idx in ids else expected)
        assert all(not torch.is_tensor(v) or v.isfinite().all() for v in value.values())
    assert all(v.isfinite().all() for v in state['model'].values())
    assert all(torch.equal(v, parent['model'][k]) for k, v in state['model'].items() if k.startswith('static.'))
    for number in [40, 80] if smoke else []:
        receipt = read(directory/f'resume_{number:04d}.json')
        assert receipt['iteration'] == number and receipt['recovered_mix'] == number/400
    # I1's smoke contains only100 steps. Its formal run has the same parent RNG
    # and a full2000-step reference, including the expanded500-step smoke span.
    reference = ROOT/'runs/pidg_staged_reconstruction/motion_integrity/formal/I1_curve_isometry/steps.jsonl'
    other = [json.loads(v) for v in reference.read_text().splitlines()][:expected]
    assert [r['frame_index'] for r in rows] == [r['frame_index'] for r in other]
    assert read(EXP/'native_preflight.json')['passed']
    assert sha(PARENT) == PARENT_SHA
    result = {'passed': True, 'actual_optimizer_updates': expected, 'physics_calls': expected//10,
              'source_parent_full_state_exact': True, 'camera_Adam_RNG_replay_exact': True, 'static_frozen_exact': True,
              'full_state_finite': True, 'camera_sequence_matches_I1': True,
              'independent_process_resume_40_and_80': smoke, 'native_preflight': read(EXP/'native_preflight.json'),
              'fully_sparse_optimizer_updates': sum(r['sparse_mix'] == 1 for r in rows),
              'checkpoint_sha256': sha(path), 'test_RGB_read': False}
    write(EXP/('preflight.json' if smoke else 'training_verification.json'), result)
    print('SPARSE_STATE_AUDIT_PASSED', expected, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native', action='store_true')
    p.add_argument('--audit-smoke', action='store_true')
    p.add_argument('--audit-formal', action='store_true')
    args = p.parse_args()
    if args.native:
        native()
    elif args.audit_smoke or args.audit_formal:
        audit(args.audit_smoke)
    else:
        p.error('Select a preflight operation')


if __name__ == '__main__':
    main()
