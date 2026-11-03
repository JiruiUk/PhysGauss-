"""Native flags-off gradient and full optimization-state audit."""
import argparse
import json
import random
import numpy as np
import torch
from ..prepare_data import ROOT, DATA, REPORT, read, write, sha
from ..motion_integrity.prepare import load_parent, PARENT, PARENT_SHA
from ..motion_integrity.train import step_loss
from .prepare import EXP, WORK
from .train import extra_loss


def native():
    torch.set_num_threads(4)
    renderer, scene, opt, _, base = load_parent()
    plan = read(EXP/'protocol.json')
    meta = read(DATA/'transforms_train.json')
    from ..train_static import camera
    from .depth_contract import verify
    contract = verify(renderer, camera(meta['frames'][0], meta, 800))
    for i, f in enumerate(meta['frames']):
        f['_index'] = i
    with np.load(ROOT/plan['graph']) as d:
        graph = tuple(torch.tensor(d[k], device='cuda') for k in ['edges','rest','weight'])
    cache = {}
    loss, _ = step_loss(scene, renderer, opt, meta['frames'][0], meta, cache, 6001, 1, True, plan, base, graph)
    loss.backward()
    gradients = {n: None if p.grad is None else p.grad.clone() for n, p in scene.named_parameters()}
    expected = float(loss.detach())
    loss, _ = step_loss(scene, renderer, opt, meta['frames'][0], meta, cache, 6001, 1, True, plan, base, graph)
    # A zero-weight branch cannot change the inherited objective or gradients.
    prior = extra_loss(scene, renderer, meta['frames'][0], meta, 1, plan, {})
    (loss+0*prior).backward()
    error, scale = 0., 0.
    for n, p in scene.named_parameters():
        old = gradients[n]
        if old is None:
            assert p.grad is None or p.grad.abs().max() == 0
        else:
            assert p.grad is not None
            error = max(error, float((p.grad-old).abs().max()))
            scale = max(scale, float(old.abs().max()))
    assert float(loss.detach()) == expected
    assert error < max(1e-7, 2e-5*scale), (error, scale)
    write(EXP/'native_preflight.json', {'passed': True, 'zero_weight_inherited_loss_exact': True,
          'gradient_max_abs_difference': error, 'gradient_scale': scale,
          'depth_backward_qualified': read(EXP/'data_audit.json'), 'parent_unchanged': sha(PARENT) == PARENT_SHA,
          'known_geometry_finite_difference': contract,
          'original_flags_off_checks': read(REPORT/'motion_integrity/flags_off.json')})
    print('ORDINAL_NATIVE_PASSED', error, flush=True)


def audit(smoke):
    plan = read(EXP/'protocol.json')
    directory = WORK/('_smoke' if smoke else 'formal')
    expected = plan['smoke_updates'] if smoke else plan['actual_updates']
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
        assert np.isfinite(row['ordinal']) and row['ordinal_weight'] == plan['ordinal_weight']*min(1.,row['local_step']/plan['ordinal_ramp'])
    assert state['rng_python'] == rng.getstate() and state['camera_stack'] == stack
    assert torch.equal(parent['rng_torch'], state['rng_torch'])
    assert all(torch.equal(a, b) for a, b in zip(parent['rng_cuda'], state['rng_cuda']))
    assert np.array_equal(parent['rng_numpy'][1], state['rng_numpy'][1])
    ids = {i for group in state['optimizer']['param_groups'] if group['name'] == 'PIDG_velocity_stress' for i in group['params']}
    for idx, value in state['optimizer']['state'].items():
        assert float(value['step']-parent['optimizer']['state'][idx]['step']) == (expected//10 if idx in ids else expected)
        assert all(not torch.is_tensor(v) or v.isfinite().all() for v in value.values())
    assert all(v.isfinite().all() for v in state['model'].values())
    assert all(torch.equal(v, parent['model'][k]) for k, v in state['model'].items() if k.startswith('static.'))
    reference = ROOT/'runs/pidg_staged_reconstruction/motion_integrity/formal/I1_curve_isometry/steps.jsonl'
    other = [json.loads(v) for v in reference.read_text().splitlines()][:expected]
    assert [r['frame_index'] for r in rows] == [r['frame_index'] for r in other]
    if smoke:
        assert all(read(directory/f'resume_{n:04d}.json')['full_state_restored'] for n in [40,80])
    assert read(EXP/'native_preflight.json')['passed'] and sha(PARENT) == PARENT_SHA
    result = {'passed': True, 'actual_optimizer_updates': expected, 'physics_calls': expected//10,
              'initial_full_state_exact': True, 'camera_Adam_RNG_replay_exact': True,
              'static_frozen_exact': True, 'full_state_finite': True,
              'full_weight_updates': sum(r['ordinal_weight'] == plan['ordinal_weight'] for r in rows),
              'camera_sequence_matches_I1': True, 'checkpoint_sha256': sha(path),
              'test_RGB_read': False, 'independent_resume40_80': smoke}
    write(EXP/('preflight.json' if smoke else 'training_verification.json'), result)
    print('ORDINAL_STATE_AUDIT_PASSED', expected, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native', action='store_true')
    p.add_argument('--audit-smoke', action='store_true')
    p.add_argument('--audit-formal', action='store_true')
    a = p.parse_args()
    if a.native:
        native()
    elif a.audit_smoke or a.audit_formal:
        audit(a.audit_smoke)
    else:
        p.error('Select operation')


if __name__ == '__main__':
    main()
