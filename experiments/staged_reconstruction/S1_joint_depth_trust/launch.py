"""Opt-in process adapter for frozen J1 trainer, independent qualification and export."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_joint_depth_trust/v1'
OUT = ROOT/'output/staged_reconstruction/S1_joint_depth_trust/v1'


def configure():
    from variants.staged_reconstruction.joint_rgb import io
    io.BASE, io.REPORT, io.RUNS, io.OUT = BASE, REPORT, RUNS, OUT
    return io


def adapt_training():
    from variants.staged_reconstruction.train_static import setup
    setup()
    from variants.staged_reconstruction.joint_rgb import runtime
    from variants.staged_reconstruction.joint_depth_trust.core import choose_runtime
    runtime.JointScene, runtime.loss = choose_runtime(runtime.JointScene, runtime.loss, enabled=True)


def worker(io, phase, case=None, arm='revisable_static', replicate=0, extras=()):
    import subprocess
    import sys
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = REPORT/f'{stamp}_{phase}_{case}_{replicate}.log'
    command = [sys.executable, '-u', str(Path(__file__)), '--enable-joint-depth-trust', '--phase', phase,
        '--arm', arm, '--replicate', str(replicate)]
    if case: command += ['--case', case]
    command += list(extras)
    with path.open('x', encoding='utf-8') as stream:
        code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    assert code == 0, f'Preserve failed log: {path}'
    return {'log': path.relative_to(ROOT).as_posix(), 'sha256': io.sha(path)}


def prepare(io):
    import json
    from variants.staged_reconstruction.prepare_data import read, sha
    assert not io.stopped() and not (REPORT/'source_freeze.json').exists()
    tests = read(REPORT/'unit_tests.json'); assert tests['passed']
    old = ROOT/'experiments/staged_reconstruction/S1_joint_rgb/accepted_recovery'
    manifest = read(old/'v1/source_freeze.json')
    for name, expected in manifest.items(): assert sha(ROOT/name) == expected, name
    original, new = read(old/'protocol.json'), read(BASE/'protocol.json')
    for key in ['parents', 'grid_args', 'net_args', 'dynamic_rates', 'Adam_eps', 'loss',
            'synthetic_promotion_gate', 'additional_gates', 'updates', 'seed', 'sampling']:
        assert original[key] == new[key], key
    io.freeze(); manifest.update(read(REPORT/'source_freeze.json'))
    for directory in [BASE, ROOT/'variants/staged_reconstruction/joint_depth_trust']:
        for path in directory.glob('*.py'): manifest[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in [ROOT/'tests/test_joint_depth_trust.py', BASE/'qualification_protocol.json']:
        manifest[path.relative_to(ROOT).as_posix()] = sha(path)
    io.atomic_json(REPORT/'source_freeze.json', manifest); io.verify()
    print('J3_TRAIN_INPUT_SOURCE_FREEZE', len(manifest), flush=True)


def native(io):
    import copy
    import numpy as np
    import torch
    from variants.staged_reconstruction.prepare_data import read, sha
    from variants.staged_reconstruction.train_static import setup
    from variants.staged_reconstruction.synthetic_audit.input import load
    from variants.staged_reconstruction.nuisance_static.runtime import calibrated_camera
    from variants.staged_reconstruction.joint_rgb.core import assert_equivalent
    from variants.staged_reconstruction.joint_depth_trust.core import choose_runtime
    torch.set_num_threads(4); renderer = setup()
    from variants.staged_reconstruction.joint_rgb.runtime import JointScene, loss
    protocol = read(BASE/'protocol.json'); q = read(BASE/'qualification_protocol.json')
    original = JointScene(protocol, 'diffuse', True)
    Protected, protected_loss = choose_runtime(JointScene, loss, True)
    scene = Protected(protocol, 'diffuse', True); scene.load_state_dict(original.state_dict())
    assert choose_runtime(JointScene, loss) == (JointScene, loss)
    assert_equivalent(original.state_dict(), scene.state_dict(), 0, 0)
    assert [n for n, _ in original.named_parameters()] == [n for n, _ in scene.named_parameters()]
    data = load(ROOT/protocol['synthetic_data']/'diffuse/train'); index = q['fixed_train_view_index']
    image = data['rgb'][index]; camera = calibrated_camera(data['views'][index], data['K'], image.shape[1], image.shape[0])
    target = torch.from_numpy(image).permute(2, 0, 1).cuda(); t = data['times'][index]
    before = {n: v.clone() for n, v in scene.state_dict().items()}
    a, b = original.render(renderer, camera, t), scene.render(renderer, camera, t)
    assert all(torch.equal(a[k], b[k]) for k in ['render', 'alpha', 'depth'])
    assert b['_self_depth_loss'].item() == 0
    loss(a, target).backward(); protected_loss(b, target).backward()
    assert_equivalent({n: p.grad for n, p in original.named_parameters()},
        {n: p.grad for n, p in scene.named_parameters()}, 1e-6, 1e-5)
    scene.zero_grad(set_to_none=True)
    with torch.no_grad():
        rows = torch.arange(len(scene.static._opacity), device='cuda', dtype=scene.static._opacity.dtype)
        scene.static._opacity[:, 0].add_(.03+.01*torch.cos(rows))
    self_value = scene.render(renderer, camera, t)['_self_depth_loss']; self_value.backward()
    grad = scene.static._opacity.grad.clone(); assert grad.isfinite().all() and grad.abs().max() > 0
    assert all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad))
        for name, p in scene.named_parameters() if name != 'static._opacity')
    row = int(grad.abs().argmax()); h = q['finite_difference_step']
    initial_value = scene.static._opacity[row, 0].detach().clone()
    finite = []
    with torch.no_grad():
        for delta in [h, -h]:
            scene.static._opacity[row, 0].copy_(initial_value+delta)
            finite.append(scene.render(renderer, camera, t)['_self_depth_loss'].item())
        scene.static._opacity[row, 0].copy_(initial_value)
    numerical, automatic = (finite[0]-finite[1])/(2*h), grad[row, 0].item()
    assert np.isclose(numerical, automatic, atol=q['finite_difference_atol'], rtol=q['finite_difference_rtol']), (numerical, automatic)
    scene.load_state_dict(before); assert_equivalent(before, scene.state_dict(), 0, 0)
    io.verify(); io.atomic_json(REPORT/'native_check.json', {'passed': True,
        'disabled_original_class_and_loss_identity': True, 'enabled_initial_RGB_alpha_D_exact': True,
        'zero_initial_self_loss_RGB_gradients_original_tolerance': True,
        'teacher_excluded_from_state_and_optimizer_parameters': True,
        'isolated_self_loss_reaches_only_static_opacity': True,
        'finite_difference': {'row': row, 'automatic': automatic, 'numerical': numerical, 'step': h,
            'atol': q['finite_difference_atol'], 'rtol': q['finite_difference_rtol']},
        'native_diagnostic_updates': 0, 'source_sha256': sha(Path(__file__))})
    print('J3_NATIVE_GRADIENT_FLAGS_OFF_PASS', automatic, numerical, flush=True)


def qualify(io):
    import torch
    from variants.staged_reconstruction.prepare_data import read, sha
    from variants.staged_reconstruction.joint_rgb.core import assert_equivalent
    assert not (REPORT/'recovery.json').exists()
    rows = []
    with io.lock('qualification.lock'):
        for case in read(BASE/'protocol.json')['cases']:
            assert not io.stopped()
            logs = [worker(io, 'train', case, extras=['--preflight-tag', 'split', '--stop-after', '20'])]
            logs += [worker(io, 'qualification-worker', case, replicate=i) for i in [0, 1]]
            directory = RUNS/'qualification'/case/'revisable_static'
            a, b = [torch.load(directory/f'replay{i}.pt', map_location='cpu', weights_only=False) for i in [0, 1]]
            assert_equivalent(a, b, 0, 0)
            rows.append({'case': case, 'arm': 'revisable_static', 'passed': True, 'logs': logs,
                'model_two_Adam_bindings_allRNG_camera_exact': True,
                'native_RGB_alpha_exact_gradients_original_tolerance': True,
                'fixed_native_gradient_Adam_constraint_update_exact': True})
        io.verify(); io.atomic_json(REPORT/'recovery.json', {'passed': True, 'rows': rows,
            'human_approved_acceptance_inherited': True, 'original_full40_failures_retained': True,
            'source_manifest_sha256': sha(REPORT/'source_freeze.json')})


def inherited_control(io, case):
    import shutil
    from variants.staged_reconstruction.prepare_data import read, sha
    p = read(BASE/'protocol.json'); old = ROOT/p['historical_control_base']/'formal'/case/'frozen_static'
    new = RUNS/'formal'/case/'frozen_static'; new.mkdir(parents=True, exist_ok=True)
    for name in ['checkpoint.pt', 'initial.pt', 'checkpoint.json']:
        assert not (new/name).exists(); shutil.copyfile(old/name, new/name)
        assert sha(old/name) == sha(new/name)
    io.atomic_json(new/'inherited_control.json', {'exact_copy_from': old.relative_to(ROOT).as_posix(),
        'historical6000_updates_not_repeated': True, 'checkpoint_sha256': sha(new/'checkpoint.pt'),
        'initial_sha256': sha(new/'initial.pt')})


def evaluate(io):
    import importlib.util
    import random
    import numpy as np
    import torch
    from variants.staged_reconstruction.prepare_data import read, sha
    from variants.staged_reconstruction.joint_rgb.core import assert_equivalent
    old_base = ROOT/'experiments/staged_reconstruction/S1_joint_rgb'
    spec = importlib.util.spec_from_file_location('j3_independent_evaluation', old_base/'evaluate.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original_audit = module.audit_checkpoint
    def checkpoint_audit(case, arm, protocol):
        if arm == 'revisable_static': return original_audit(case, arm, protocol)
        directory = RUNS/'formal'/case/arm
        control = read(directory/'inherited_control.json'); old = ROOT/control['exact_copy_from']
        assert sha(directory/'checkpoint.pt') == sha(old/'checkpoint.pt') == control['checkpoint_sha256']
        assert sha(directory/'initial.pt') == sha(old/'initial.pt') == control['initial_sha256']
        final = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
        parent = torch.load(ROOT/protocol['parents'][case]['checkpoint'], map_location='cpu', weights_only=False)
        accepted = ROOT/'experiments/staged_reconstruction/S1_joint_rgb/accepted_recovery'
        assert final['protocol_sha256'] == sha(accepted/'protocol.json')
        assert final['source_sha256'] == sha(accepted/'v1/source_freeze.json')
        assert final['parent_sha256'] == protocol['parents'][case]['sha256'] and final['iteration'] == 6000
        for name, value in parent['model'].items(): assert torch.equal(value, final['model']['static.'+name])
        assert_equivalent(parent['optimizer'], final['optimizers'][0], 0, 0)
        rng = random.Random(); rng.setstate(parent['rng_python']); stack = list(parent['camera_stack']); trace = []
        for _ in range(6000):
            if not stack: stack = rng.sample(range(96), 96)
            trace.append(stack.pop())
        assert final['sample_trace'] == trace and final['camera_stack'] == stack
        assert final['rng_python'] == rng.getstate() and final['visits'] == np.bincount(trace, minlength=96).tolist()
        result = {'passed': True, 'case': case, 'arm': arm, 'iteration': 6000,
            'checkpoint_sha256': sha(directory/'checkpoint.pt'), 'initial_sha256': sha(directory/'initial.pt'),
            'inherited_historical_control_exact': True, 'full6000_camera_RNG_and_static_Adam_exact': True}
        io.atomic_json(REPORT/f'formal_{case}_{arm}_audit.json', result)
        return final, result
    module.audit_checkpoint = checkpoint_audit; module.main()
    io.atomic_json(REPORT/'evaluation_adapter.json', {'source_sha256': sha(Path(__file__)),
        'original_evaluator_sha256': sha(old_base/'evaluate.py'), 'original_quality_gates_unchanged': True,
        'only_inherited_control_provenance_audit_replaced': True})


def export(io):
    import importlib.util
    spec = importlib.util.spec_from_file_location('j3_native_exports', ROOT/'experiments/staged_reconstruction/S1_joint_rgb/accepted_recovery/export.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.BASE, module.REPORT, module.RUNS = BASE, REPORT, RUNS
    module.configure = lambda: io
    module.main()


def pipeline(io):
    import os
    from variants.staged_reconstruction.prepare_data import read
    assert read(REPORT/'recovery.json')['passed']
    with io.lock('pipeline.lock'):
        state = {'status': 'training', 'pid': os.getpid(), 'completed': []}; io.atomic_json(RUNS/'pipeline.json', state)
        try:
            for case in read(BASE/'protocol.json')['cases']:
                if io.stopped():
                    state['status'] = 'saved_stopped'; io.atomic_json(RUNS/'pipeline.json', state); return
                if not (RUNS/'formal'/case/'frozen_static/inherited_control.json').exists(): inherited_control(io, case)
                receipt = RUNS/'formal'/case/'revisable_static/checkpoint.json'
                if not (receipt.exists() and read(receipt)['iteration'] == 6000):
                    state['completed'].append(worker(io, 'train', case))
                if io.stopped():
                    state['status'] = 'saved_stopped'; io.atomic_json(RUNS/'pipeline.json', state); return
            state['status'] = 'evaluating'; io.atomic_json(RUNS/'pipeline.json', state)
            if not (REPORT/'synthetic_evaluation.json').exists(): state['evaluation'] = worker(io, 'evaluate')
            if not (REPORT/'exports.json').exists(): state['exports'] = worker(io, 'export')
            state['status'] = 'evaluated_exported_pending_manual_review'; io.atomic_json(RUNS/'pipeline.json', state)
        except Exception as error:
            state.update(status='failed', error=repr(error)); io.atomic_json(RUNS/'pipeline.json', state); raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-joint-depth-trust', action='store_true')
    parser.add_argument('--phase', choices=['prepare', 'native', 'qualify', 'qualification-worker', 'train', 'evaluate', 'export', 'pipeline'], default='prepare')
    parser.add_argument('--case', choices=['diffuse', 'glossy'], default='diffuse')
    parser.add_argument('--arm', choices=['revisable_static'], default='revisable_static')
    parser.add_argument('--replicate', type=int, choices=[0, 1], default=0)
    parser.add_argument('--preflight-tag', choices=['split']); parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if not args.enable_joint_depth_trust:
        print('J3_DISABLED: no model imports or checkpoint access'); return
    io = configure(); assert not io.stopped()
    if args.phase == 'prepare': prepare(io); return
    io.verify()
    if args.phase == 'native': native(io)
    elif args.phase == 'qualify': qualify(io)
    elif args.phase == 'pipeline': pipeline(io)
    elif args.phase == 'evaluate': evaluate(io)
    elif args.phase == 'export': export(io)
    elif args.phase in ['train', 'qualification-worker']:
        adapt_training()
        if args.phase == 'qualification-worker':
            from experiments.staged_reconstruction.S1_joint_rgb.accepted_recovery import launch as old
            old.BASE, old.REPORT, old.RUNS = BASE, REPORT, RUNS
            old.qualification_worker(io, args)
        else:
            import sys
            from variants.staged_reconstruction.joint_rgb.train import main as train
            sys.argv = [__file__, '--enable-joint-rgb', '--case', args.case, '--arm', args.arm]
            if args.preflight_tag: sys.argv += ['--preflight-tag', args.preflight_tag]
            if args.stop_after: sys.argv += ['--stop-after', str(args.stop_after)]
            train()


if __name__ == '__main__': main()
