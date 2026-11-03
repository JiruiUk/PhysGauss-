"""Full-state opt-in training, one frozen/revisable static switch, RGB loss only."""
import argparse
import json
import os
import random
import time
from pathlib import Path
import cv2
import torch
from ..prepare_data import ROOT, read, sha
from ..synthetic_audit.input import load
from ..train_static import setup
from ..nuisance_static.runtime import calibrated_camera
from .core import rng_state, restore_rng
from .io import BASE, REPORT, RUNS, atomic_json, verify, stopped, lock


def save(directory, scene, optimizers, step, stack, visits, trace, case, arm):
    state = {'model': scene.state_dict(), 'optimizers': [o.state_dict() for o in optimizers],
        'iteration': step, 'camera_stack': list(stack), 'visits': list(visits),
        'sample_trace': list(trace), 'case': case, 'arm': arm,
        'protocol_sha256': sha(BASE/'protocol.json'), 'source_sha256': sha(REPORT/'source_freeze.json'),
        'parent_sha256': read(BASE/'protocol.json')['parents'][case]['sha256'],
        'point_budget_not_body_count': True, **rng_state()}
    temp = directory/'checkpoint.tmp'; torch.save(state, temp)
    os.replace(temp, directory/'checkpoint.pt')
    atomic_json(directory/'checkpoint.json', {'iteration': step, 'sha256': sha(directory/'checkpoint.pt')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-joint-rgb', action='store_true')
    parser.add_argument('--case', choices=['diffuse', 'glossy'], default='diffuse')
    parser.add_argument('--arm', choices=['frozen_static', 'revisable_static'], default='frozen_static')
    parser.add_argument('--preflight-tag', choices=['full', 'split'])
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if not args.enable_joint_rgb:
        raise SystemExit('J1 is disabled; pass --enable-joint-rgb')
    assert not stopped(), 'User stop request; never auto-resume'
    verify(); assert read(REPORT/'unit_tests.json')['passed']
    assert read(REPORT/'native_check.json')['passed']
    if args.preflight_tag is None:
        assert read(REPORT/'recovery.json')['passed']
    with lock('training.lock'):
        directory = RUNS/('preflight/'+args.preflight_tag if args.preflight_tag else 'formal')/args.case/args.arm
        directory.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(4); cv2.setNumThreads(2)
        protocol = read(BASE/'protocol.json'); renderer = setup()
        from .runtime import JointScene, loss
        scene = JointScene(protocol, args.case, args.arm == 'revisable_static')
        optimizers = scene.optimizers(protocol)
        values = load(ROOT/protocol['synthetic_data']/args.case/'train')
        images, views, times = values['rgb'], values['views'], values['times']
        assert len(images) == 96
        height, width = images.shape[1:3]
        cameras = [calibrated_camera(v, values['K'], width, height) for v in views]
        targets = torch.from_numpy(images).permute(0, 3, 1, 2).cuda()
        limit = protocol['recovery_full_steps'] if args.preflight_tag else protocol['updates']
        if not (directory/'checkpoint.pt').exists():
            restore_rng(scene.parent)
            save(directory, scene, optimizers, 0, scene.parent['camera_stack'], [0]*96, [], args.case, args.arm)
            (directory/'initial.pt').write_bytes((directory/'checkpoint.pt').read_bytes())
        assert sha(directory/'checkpoint.pt') == read(directory/'checkpoint.json')['sha256']
        state = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
        assert state['case'] == args.case and state['arm'] == args.arm
        assert state['source_sha256'] == sha(REPORT/'source_freeze.json')
        assert state['protocol_sha256'] == sha(BASE/'protocol.json')
        assert state['parent_sha256'] == protocol['parents'][args.case]['sha256']
        scene.load_state_dict(state['model'])
        for optimizer, payload in zip(optimizers, state['optimizers']):
            optimizer.load_state_dict(payload)
        restore_rng(state); stack, visits, trace = state['camera_stack'], state['visits'], state['sample_trace']
        assert state['iteration'] < limit, 'Do not repeat completed training'
        frozen_geometry = {name: getattr(scene.static, name).detach().clone() for name in ['_xyz', '_scaling', '_rotation']}
        began = time.monotonic(); completed = state['iteration']
        atomic_json(RUNS/'state.json', {'status': 'training', 'pid': os.getpid(), 'case': args.case,
            'arm': args.arm, 'iteration': completed, 'preflight_tag': args.preflight_tag})
        try:
            for step in range(completed+1, limit+1):
                if not stack:
                    stack = random.sample(range(96), 96)
                index = stack.pop()
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                package = scene.render(renderer, cameras[index], times[index])
                value = loss(package, targets[index]); assert value.isfinite()
                value.backward()
                assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
                for optimizer in optimizers:
                    optimizer.step()
                scene.constrain(); visits[index] += 1; trace.append(index); completed = step
                with (directory/'losses.jsonl').open('a') as stream:
                    stream.write(json.dumps({'step': step, 'view': index, 'loss': float(value.detach())})+'\n')
                if step % 100 == 0:
                    assert all(torch.equal(value, getattr(scene.static, name)) for name, value in frozen_geometry.items())
                    atomic_json(RUNS/'state.json', {'status': 'training', 'pid': os.getpid(), 'case': args.case,
                        'arm': args.arm, 'iteration': step, 'preflight_tag': args.preflight_tag})
                    print('J1_TRAIN', args.case, args.arm, step, float(value.detach()), round(time.monotonic()-began, 1), flush=True)
                if step % protocol['checkpoint_interval'] == 0:
                    save(directory, scene, optimizers, step, stack, visits, trace, args.case, args.arm)
                if stopped() or args.stop_after and step >= args.stop_after:
                    save(directory, scene, optimizers, step, stack, visits, trace, args.case, args.arm)
                    atomic_json(RUNS/'state.json', {'status': 'saved_stopped' if stopped() else 'preflight_split_saved',
                        'iteration': step, 'case': args.case, 'arm': args.arm}); return
            save(directory, scene, optimizers, limit, stack, visits, trace, args.case, args.arm); verify()
            atomic_json(RUNS/'state.json', {'status': 'trained_pending_evaluation', 'case': args.case,
                'arm': args.arm, 'iteration': limit, 'preflight_tag': args.preflight_tag})
            print('J1_TRAIN_COMPLETE', args.case, args.arm, limit, flush=True)
        except Exception as error:
            atomic_json(directory/'failure.json', {'error': repr(error), 'last_completed_in_memory': completed,
                'saved_state': 'last atomic checkpoint; do not treat failed partial step as saved'})
            atomic_json(RUNS/'state.json', {'status': 'failed', 'case': args.case, 'arm': args.arm, 'error': repr(error)})
            raise


if __name__ == '__main__':
    main()
