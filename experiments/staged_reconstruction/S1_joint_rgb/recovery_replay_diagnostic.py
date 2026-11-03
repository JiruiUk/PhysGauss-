"""Independent diagnostic only; never changes failed40 gate or authorizes training."""
import argparse
import subprocess
import sys
from pathlib import Path
import torch
from variants.staged_reconstruction.prepare_data import ROOT, read, sha
from variants.staged_reconstruction.joint_rgb.io import BASE, REPORT, RUNS, atomic_json, verify, stopped, lock
from variants.staged_reconstruction.joint_rgb.core import assert_equivalent, rng_state, restore_rng

DIRECTORY = REPORT/'recovery_replay_diagnostic'
SAVED = RUNS/'preflight/split/diffuse/frozen_static/split20.pt'


def cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu(v) for v in value)
    return value


def loaded():
    from variants.staged_reconstruction.train_static import setup
    renderer = setup()
    from variants.staged_reconstruction.joint_rgb.runtime import JointScene
    protocol = read(BASE/'protocol.json'); state = torch.load(SAVED, map_location='cpu', weights_only=False)
    scene = JointScene(protocol, 'diffuse'); optimizers = scene.optimizers(protocol)
    scene.load_state_dict(state['model'])
    for opt, payload in zip(optimizers, state['optimizers']):
        opt.load_state_dict(payload)
    assert_equivalent(state['model'], cpu(scene.state_dict()), 0, 0)
    assert_equivalent(state['optimizers'], cpu([opt.state_dict() for opt in optimizers]), 0, 0)
    named = {id(p): name for name, p in scene.named_parameters()}
    bindings = [[[named[id(p)] for p in group['params']] for group in opt.param_groups] for opt in optimizers]
    expected_static = [[group['name']] for group in state['optimizers'][0]['param_groups']]
    assert bindings[0] == [['static.'+name for name in group] for group in expected_static]
    flattened = [name for opts in bindings for group in opts for name in group]
    assert len(flattened) == len(set(flattened))
    restore_rng(state)
    return renderer, scene, optimizers, state, bindings


def worker(phase, replicate):
    renderer, scene, optimizers, state, bindings = loaded()
    if phase == 'fixture':
        from variants.staged_reconstruction.synthetic_audit.input import load
        from variants.staged_reconstruction.nuisance_static.runtime import calibrated_camera
        from variants.staged_reconstruction.joint_rgb.runtime import loss
        values = load(ROOT/read(BASE/'protocol.json')['synthetic_data']/'diffuse/train')
        index = state['camera_stack'][-1]; image = values['rgb'][index]
        camera = calibrated_camera(values['views'][index], values['K'], image.shape[1], image.shape[0])
        target = torch.from_numpy(image).permute(2, 0, 1).cuda()
        value = loss(scene.render(renderer, camera, values['times'][index]), target); value.backward()
        gradients = {name: cpu(p.grad) for name, p in scene.named_parameters() if p.grad is not None}
        assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
        torch.save({'gradients': gradients, 'bindings': bindings, 'step20_sha256': sha(SAVED),
            'source_sha256': sha(Path(__file__)), 'next_fixed_view': index,
            'native_full_RGB_loss': float(value.detach()), 'use': 'optimizer serialization diagnostic only; never a training asset'}, DIRECTORY/'fixed_native_gradients.pt')
        return
    fixture = torch.load(DIRECTORY/'fixed_native_gradients.pt', map_location='cpu', weights_only=False)
    assert fixture['step20_sha256'] == sha(SAVED) and fixture['source_sha256'] == sha(Path(__file__))
    assert bindings == fixture['bindings']
    for name, p in scene.named_parameters():
        p.grad = fixture['gradients'][name].to(p.device).clone() if name in fixture['gradients'] else None
    for optimizer in optimizers:
        optimizer.step()
    # No train trajectory, camera advancement or constraint is run. This is an
    # isolated Adam algebra check on disposable copies, preserving all states.
    torch.save({'model': cpu(scene.state_dict()), 'optimizers': cpu([o.state_dict() for o in optimizers]),
        'all_RNG': rng_state(), 'camera_stack': state['camera_stack'], 'visits': state['visits'],
        'sample_trace': state['sample_trace'], 'optimizer_parameter_bindings': bindings,
        'no_train_sampling_or_checkpoint_update': True}, DIRECTORY/f'replay{replicate}_values.pt')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-recovery-replay-diagnostic', action='store_true')
    parser.add_argument('--phase', choices=['fixture', 'replay']); parser.add_argument('--replicate', type=int, default=0)
    args = parser.parse_args()
    if not args.enable_recovery_replay_diagnostic:
        raise SystemExit('Recovery replay diagnostic is disabled')
    assert not stopped(); verify(); torch.set_num_threads(4)
    if args.phase:
        worker(args.phase, args.replicate); return
    assert not (DIRECTORY/'review.json').exists()
    DIRECTORY.mkdir(exist_ok=True)
    with lock('replay_diagnostic.lock'):
        records = []
        for phase, replicate in [('fixture', 0), ('replay', 0), ('replay', 1)]:
            command = [sys.executable, '-u', __file__, '--enable-recovery-replay-diagnostic', '--phase', phase,
                '--replicate', str(replicate)]
            log = DIRECTORY/f'{phase}{replicate}.log'
            with log.open('x') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            assert result.returncode == 0, str(log)
            records.append({'command': command, 'log': log.relative_to(ROOT).as_posix(), 'sha256': sha(log)})
        a, b = [torch.load(DIRECTORY/f'replay{i}_values.pt', map_location='cpu', weights_only=False) for i in range(2)]
        assert_equivalent(a, b, 0, 0); verify()
        atomic_json(DIRECTORY/'review.json', {'passed': True, 'independent_processes': 2,
            'immediate_model_both_Adam_and_parameter_binding_exact': True,
            'identical_saved_native_gradients_produce_exact_model_and_both_Adam_update': True,
            'all_RNG_camera_stack_visits_trace_exact': True,
            'original_full40_gate_still_failed': True, 'formal_training_admitted': False,
            'requires_human_selection_to_change_recovery_acceptance': True,
            'source_sha256': sha(Path(__file__)), 'checkpoint_sha256': sha(SAVED),
            'fixed_native_gradients_sha256': sha(DIRECTORY/'fixed_native_gradients.pt'), 'commands': records})
        print('J1_EXACT_SAVED_NATIVE_GRADIENT_ADAM_REPLAY_DIAGNOSTIC_PASSED_NO_TRAINING_ADMISSION', flush=True)


if __name__ == '__main__':
    main()
