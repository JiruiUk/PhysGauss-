"""Explicit accepted-recovery paths; unchanged frozen J1 model/training code."""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_joint_rgb/accepted_recovery'


def configure():
    from variants.staged_reconstruction.joint_rgb import io
    from variants.staged_reconstruction.prepare_data import read, sha
    original = ROOT/'experiments/staged_reconstruction/S1_joint_rgb/v1/source_freeze.json'
    for name, expected in read(original).items():
        assert sha(ROOT/name) == expected, name
    io.BASE, io.REPORT, io.RUNS = BASE, REPORT, RUNS
    io.OUT = ROOT/'output/staged_reconstruction/S1_joint_rgb/accepted_recovery'
    return io


def command(io, phase, case=None, arm=None, replicate=None, extra=()):
    values = [sys.executable, '-u', str(Path(__file__)), '--enable-accepted-joint-rgb', '--phase', phase]
    for key, value in [('--case', case), ('--arm', arm), ('--replicate', replicate)]:
        if value is not None: values += [key, str(value)]
    values += list(extra)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log = REPORT/f'{stamp}_{phase}_{case}_{arm}_{replicate}.log'
    with log.open('x') as stream:
        result = subprocess.run(values, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    assert result.returncode == 0, 'Preserve failed log '+str(log)
    return {'command': values, 'log': log.relative_to(ROOT).as_posix(), 'sha256': io.sha(log)}


def prepare(io):
    from variants.staged_reconstruction.prepare_data import read, sha
    assert not io.stopped() and not (REPORT/'source_freeze.json').exists()
    source = ROOT/'experiments/staged_reconstruction/S1_joint_rgb'
    old, new = read(source/'protocol.json'), read(BASE/'protocol.json')
    for key, value in old.items():
        if key != 'stage': assert new[key] == value, key
    assert new['Adam_eps'] == 1e-15 and new['human_approved_recovery_acceptance']
    assert read(BASE/'authorization.json')['answer'] == '采用精确状态恢复验收（推荐）'
    tests = read(REPORT/'launcher_tests.json'); assert tests['passed']
    manifest = read(source/'v1/source_freeze.json').copy()
    files = list(BASE.glob('*.py'))+[BASE/'plan.md', BASE/'protocol.json', BASE/'preregistration.json',
        BASE/'authorization.json', ROOT/'tests/test_accepted_joint_rgb.py', source/'evaluate.py',
        source/'v1/native_check.json', source/'v1/unit_tests.json', source/'v1/recovery_failure_attempt1.json',
        source/'v1/recovery_diagnostic.json']
    for path in files: manifest[path.relative_to(ROOT).as_posix()] = sha(path)
    io.atomic_json(REPORT/'source_freeze.json', manifest)
    io.atomic_json(REPORT/'unit_tests.json', {'passed': True, 'original_core_tests': 6,
        'original_receipt_sha256': sha(source/'v1/unit_tests.json'), 'launcher_receipt_sha256': sha(REPORT/'launcher_tests.json')})
    assert read(source/'v1/native_check.json')['passed']
    io.atomic_json(REPORT/'native_check.json', {'passed': True, 'inherited_unchanged_native_code_and_configuration': True,
        'original_receipt_sha256': sha(source/'v1/native_check.json'), 'all_original_native_parameters_unchanged': True})
    io.verify(); print('J1_A_FRESH_SOURCE_INPUT_FREEZE', len(manifest), flush=True)


def cpu(value):
    import torch
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu(v) for v in value)
    return value


def qualification_worker(io, args):
    import torch
    from variants.staged_reconstruction.joint_rgb.core import assert_equivalent, restore_rng, rng_state
    from variants.staged_reconstruction.prepare_data import read, sha
    from variants.staged_reconstruction.train_static import setup
    from variants.staged_reconstruction.synthetic_audit.input import load
    from variants.staged_reconstruction.nuisance_static.runtime import calibrated_camera
    torch.set_num_threads(4); renderer = setup()
    from variants.staged_reconstruction.joint_rgb.runtime import JointScene, loss
    protocol = read(BASE/'protocol.json')
    saved = RUNS/'preflight/split'/args.case/args.arm/'checkpoint.pt'
    state = torch.load(saved, map_location='cpu', weights_only=False); assert state['iteration'] == 20
    assert state['protocol_sha256'] == sha(BASE/'protocol.json') and state['source_sha256'] == sha(REPORT/'source_freeze.json')
    scene = JointScene(protocol, args.case, args.arm == 'revisable_static'); opts = scene.optimizers(protocol)
    scene.load_state_dict(state['model'])
    for opt, payload in zip(opts, state['optimizers']): opt.load_state_dict(payload)
    assert_equivalent(state['model'], cpu(scene.state_dict()), 0, 0)
    assert_equivalent(state['optimizers'], cpu([o.state_dict() for o in opts]), 0, 0)
    names = {id(p): name for name, p in scene.named_parameters()}
    bindings = [[[names[id(p)] for p in group['params']] for group in opt.param_groups] for opt in opts]
    assert bindings[0] == [['static.'+group['name']] for group in state['optimizers'][0]['param_groups']]
    flat = [name for opt in bindings for group in opt for name in group]; assert len(flat) == len(set(flat))
    restore_rng(state)
    expected_rng = {k: state[k] for k in ['rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda']}
    assert_equivalent(expected_rng, rng_state(), 0, 0)
    values = load(ROOT/protocol['synthetic_data']/args.case/'train'); index = state['camera_stack'][-1]
    image = values['rgb'][index]; camera = calibrated_camera(values['views'][index], values['K'], image.shape[1], image.shape[0])
    package = scene.render(renderer, camera, values['times'][index])
    target = torch.from_numpy(image).permute(2, 0, 1).cuda(); value = loss(package, target)
    assert value.isfinite(); value.backward()
    gradients = {name: cpu(p.grad) for name, p in scene.named_parameters() if p.grad is not None}
    assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
    directory = RUNS/'qualification'/args.case/args.arm; directory.mkdir(parents=True, exist_ok=True)
    fixture_path = directory/'fixed_native_gradients.pt'
    fixture = {'gradients': gradients, 'rgb': cpu(package['render']), 'alpha': cpu(package['alpha']),
        'bindings': bindings, 'checkpoint_sha256': sha(saved), 'source_sha256': sha(Path(__file__)), 'view': index}
    if args.replicate == 0:
        assert not fixture_path.exists(); torch.save(fixture, fixture_path)
    else:
        actual = fixture; fixture = torch.load(fixture_path, map_location='cpu', weights_only=False)
        assert fixture['checkpoint_sha256'] == sha(saved) and fixture['source_sha256'] == sha(Path(__file__))
        assert fixture['bindings'] == bindings and fixture['view'] == index
        assert torch.equal(actual['rgb'], fixture['rgb']) and torch.equal(actual['alpha'], fixture['alpha'])
        assert_equivalent(actual['gradients'], fixture['gradients'], protocol['recovery_atol'], protocol['recovery_rtol'])
    for name, p in scene.named_parameters():
        p.grad = fixture['gradients'][name].to(p.device).clone() if name in fixture['gradients'] else None
    for opt in opts: opt.step()
    scene.constrain()
    assert_equivalent(expected_rng, rng_state(), 0, 0)
    torch.save({'model': cpu(scene.state_dict()), 'optimizers': cpu([o.state_dict() for o in opts]),
        'all_RNG': rng_state(), 'camera_stack': state['camera_stack'], 'visits': state['visits'],
        'sample_trace': state['sample_trace'], 'bindings': bindings}, directory/f'replay{args.replicate}.pt')
    io.verify(); print('J1_A_NATIVE_STATE_AND_GRADIENT_REPLAY', args.case, args.arm, args.replicate, flush=True)


def qualify(io):
    import torch
    from variants.staged_reconstruction.joint_rgb.core import assert_equivalent
    from variants.staged_reconstruction.prepare_data import read, sha
    assert not (REPORT/'recovery.json').exists(); io.verify(); torch.set_num_threads(4)
    protocol = read(BASE/'protocol.json'); rows = []
    with io.lock('qualification.lock'):
        for case in protocol['cases']:
            for arm in protocol['arms']:
                assert not io.stopped()
                logs = [command(io, 'train', case, arm, extra=['--preflight-tag', 'split', '--stop-after', '20'])]
                logs += [command(io, 'qualification-worker', case, arm, replicate=i) for i in range(2)]
                directory = RUNS/'qualification'/case/arm
                a, b = [torch.load(directory/f'replay{i}.pt', map_location='cpu', weights_only=False) for i in range(2)]
                assert_equivalent(a, b, 0, 0)
                state = torch.load(RUNS/'preflight/split'/case/arm/'checkpoint.pt', map_location='cpu', weights_only=False)
                assert a['camera_stack'] == state['camera_stack'] and a['visits'] == state['visits'] and a['sample_trace'] == state['sample_trace']
                rows.append({'case': case, 'arm': arm, 'passed': True, 'logs': logs,
                    'step20_sha256': sha(RUNS/'preflight/split'/case/arm/'checkpoint.pt'),
                    'fixed_native_gradients_sha256': sha(directory/'fixed_native_gradients.pt'),
                    'both_raw_native_RGB_alpha_exact_and_gradients_original_tolerance': True,
                    'model_both_Adam_bindings_all_RNG_camera_exact': True,
                    'fixed_gradient_original_Adam_and_constraint_replay_exact': True})
                print('J1_A_RECOVERY_QUALIFIED', case, arm, flush=True)
        io.verify(); io.atomic_json(REPORT/'recovery.json', {'passed': True, 'rows': rows,
            'acceptance': protocol['recovery_acceptance'], 'all_four_required': True,
            'authorization_sha256': sha(BASE/'authorization.json'), 'source_manifest_sha256': sha(REPORT/'source_freeze.json'),
            'original_full40_gate_still_failed': True, 'original_epsilon_1e15_retained': True,
            'raw_gradient_atol': protocol['recovery_atol'], 'raw_gradient_rtol': protocol['recovery_rtol'],
            'quality_gates_unchanged': True, 'source_sha256': sha(Path(__file__))})


def pipeline(io):
    from variants.staged_reconstruction.prepare_data import read, sha
    io.verify(); assert read(REPORT/'recovery.json')['passed']
    protocol = read(BASE/'protocol.json')
    with io.lock('pipeline.lock'):
        state = {'status': 'training', 'pid': os.getpid(), 'completed': [], 'source_sha256': sha(Path(__file__))}
        io.atomic_json(RUNS/'pipeline.json', state)
        try:
            for case in protocol['cases']:
                for arm in protocol['arms']:
                    if io.stopped():
                        state['status'] = 'saved_stopped'; io.atomic_json(RUNS/'pipeline.json', state); return
                    state.update(case=case, arm=arm); io.atomic_json(RUNS/'pipeline.json', state)
                    receipt = RUNS/'formal'/case/arm/'checkpoint.json'
                    if receipt.exists() and read(receipt)['iteration'] == protocol['updates']:
                        assert sha(receipt.with_name('checkpoint.pt')) == read(receipt)['sha256']
                    else:
                        state['completed'].append(command(io, 'train', case, arm))
                        io.atomic_json(RUNS/'pipeline.json', state)
                    if io.stopped():
                        state['status'] = 'saved_stopped'; io.atomic_json(RUNS/'pipeline.json', state); return
            state.update(status='evaluating'); io.atomic_json(RUNS/'pipeline.json', state)
            if not (REPORT/'synthetic_evaluation.json').exists():
                state['evaluation'] = command(io, 'evaluate')
            state.update(status='evaluated_pending_exports_and_manual_review')
            io.atomic_json(RUNS/'pipeline.json', state)
            print('J1_A_TRAINED_EVALUATED_PENDING_EXPORT_REVIEW', flush=True)
        except Exception as error:
            state.update(status='failed', error=repr(error)); io.atomic_json(RUNS/'pipeline.json', state); raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-accepted-joint-rgb', action='store_true')
    parser.add_argument('--phase', choices=['prepare', 'qualify', 'qualification-worker', 'train', 'pipeline', 'evaluate'], default='prepare')
    parser.add_argument('--case', choices=['diffuse', 'glossy'], default='diffuse')
    parser.add_argument('--arm', choices=['frozen_static', 'revisable_static'], default='frozen_static')
    parser.add_argument('--replicate', type=int, choices=[0, 1], default=0)
    parser.add_argument('--preflight-tag', choices=['split']); parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if not args.enable_accepted_joint_rgb: raise SystemExit('Accepted J1 experiment is disabled')
    io = configure(); assert not io.stopped()
    if args.phase == 'prepare': prepare(io); return
    io.verify()
    if args.phase == 'qualify': qualify(io)
    elif args.phase == 'qualification-worker': qualification_worker(io, args)
    elif args.phase == 'pipeline': pipeline(io)
    elif args.phase == 'evaluate':
        runpy.run_path(str(BASE.parent/'evaluate.py'), run_name='__main__')
    elif args.phase == 'train':
        from variants.staged_reconstruction.joint_rgb.train import main as train
        sys.argv = [__file__, '--enable-joint-rgb', '--case', args.case, '--arm', args.arm]
        if args.preflight_tag: sys.argv += ['--preflight-tag', args.preflight_tag]
        if args.stop_after: sys.argv += ['--stop-after', str(args.stop_after)]
        train()


if __name__ == '__main__': main()
