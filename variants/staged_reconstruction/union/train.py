"""Separate opt-in count-free warmup and original S3 physics continuation."""
import argparse
import json
import os
import random
import shutil
import time
import numpy as np
import torch
from ..prepare_data import ROOT, REPORT, RUNS, DATA, read, write, sha
from ..train_static import setup, camera
from ..stage2.train import image, objective
from .core import dynamic_mask, union_coverage
from .prepare import UNION_REPORT, UNION_RUNS, verify_sources
from .runtime import save, restore


def reconstruction(package, gt, rgba, scene, cam):
    color = objective(package['render'], gt)
    mask = torch.as_tensor(dynamic_mask(rgba), device=gt.device)
    r, g, b = gt
    static = (b > r*1.3+.02) & (b > g*1.05)
    foreground = (package['render'][:, mask]-gt[:, mask]).abs().mean() if mask.any() else color*0
    static_rgb = (package['render'][:, static]-gt[:, static]).abs().mean() if static.any() else color*0
    coverage = union_coverage(package['means3D_deform'][len(scene.static.get_xyz):], cam, mask)
    return color+.5*foreground+.5*static_rgb+2.*coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-union-stage', action='store_true')
    parser.add_argument('--phase', choices=['warmup', 'physics'], default='warmup')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if not args.enable_union_stage:
        raise SystemExit('S3-U is opt-in; pass --enable-union-stage')
    verify_sources()
    protocol = read(UNION_REPORT/'protocol.json')
    protocol_hash = sha(UNION_REPORT/'protocol.json')
    torch.set_num_threads(4)
    random.seed(protocol['seed'])
    np.random.seed(protocol['seed'])
    torch.manual_seed(protocol['seed'])
    directory = UNION_RUNS/args.phase/('_smoke' if args.smoke else 'formal')
    directory.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = open(UNION_RUNS/'training.lock', 'a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit('S3-U already running; no duplicate launched')
    try:
        if (UNION_RUNS/'stop.request').exists():
            raise SystemExit('Union stop request present; do not resume paused work')
        if args.phase == 'physics':
            admission = read(UNION_REPORT/'warmup_admission.json')
            parent = UNION_RUNS/'warmup/formal/checkpoint.pt'
            assert admission['admitted_for_conditional_physics'] and admission['native_images_reviewed']
            assert sha(parent) == admission['checkpoint_sha256']
            parent_hash = sha(parent)
        else:
            parent = None
            parent_hash = protocol['static_checkpoint_sha256']
        if not args.smoke:
            assert read(UNION_REPORT/f'{args.phase}_smoke_verification.json')['passed']
        renderer = setup()
        from .model import UnionScene
        scene = UnionScene(protocol)
        optimizer = scene.optimizer()
        saved = directory/'checkpoint.pt'
        if saved.exists():
            state = restore(saved, scene, optimizer, protocol_hash)
            assert state['phase'] == args.phase and state['parent_sha256'] == parent_hash
            start, stack = state['iteration'], state['camera_stack']
            print('UNION_FULL_STATE_RESTORED', args.phase, start, flush=True)
        else:
            if parent is not None:
                previous = restore(parent, scene, optimizer, protocol_hash)
                assert previous['phase'] == 'warmup' and previous['iteration'] == protocol['warmup_updates']
                stack = previous['camera_stack']
            else:
                stack = []
            start = 0
            save(directory, scene, optimizer, 0, stack, args.phase, protocol_hash, parent_hash)
            shutil.copyfile(saved, directory/'initial.pt')
            write(directory/'initial.json', {'sha256': sha(directory/'initial.pt'), 'phase': args.phase,
                'trainable_parameters': sum(p.numel() for p in scene.parameters() if p.requires_grad),
                'dynamic_parameters': sum(p.numel() for p in scene.pools.parameters()),
                'grid_parameters': sum(p.numel() for p in scene.grids.parameters()),
                'deform_parameters': sum(p.numel() for p in scene.networks.parameters()),
                'velocity_stress_parameters': sum(p.numel() for p in scene.physics.parameters()), 'physical_body_number': None})
        frozen = {k: v.detach().clone() for k, v in scene.static.state_dict().items()}
        meta = read(DATA/'transforms_train.json')
        cache = {}
        updates = protocol[args.phase+'_updates']
        completed = start
        began = time.monotonic()
        from variants.momentum_guard.balance import balance
        for step in range(start+1, updates+1):
            if not stack:
                stack = random.sample(range(len(meta['frames'])), len(meta['frames']))
            index = stack.pop()
            frame = meta['frames'][index]
            width = 1600 if args.phase == 'physics' or step > 12000 else 800
            key = (index, width)
            if key not in cache:
                rgba = image(frame, width)
                cache[key] = (rgba, torch.tensor(rgba[..., :3]*rgba[..., 3:4]).permute(2, 0, 1), camera(frame, meta, width))
            rgba, cpu_gt, cam = cache[key]
            gt = cpu_gt.cuda()
            optimizer.zero_grad(set_to_none=True)
            package = scene.render(renderer, cam, frame['time'])
            rgb = reconstruction(package, gt, rgba, scene, cam)
            loss = rgb
            if args.phase == 'physics' and step % protocol['physics_every'] == 0:
                terms, rows = scene.physics.loss(scene, frame['time'], step, protocol['physics_samples_global'], protocol['time_step'])
                physical = protocol['kinematic_weight']*terms['kinematic']+protocol['CMR_weight']*terms['CMR']
                groups = {'grid': list(scene.grids.parameters()), 'deformation': list(scene.networks.parameters()), 'canonical': [scene.pools[0]._xyz]}
                guarded, guard = balance(rgb, physical, groups, list(scene.physics.parameters()), protocol['shared_gradient_cap'], 1.)
                loss = loss+guarded
                with (directory/'physics.jsonl').open('a') as stream:
                    stream.write(json.dumps({'step': step, 'frame': frame['file_path'],
                        'terms': {k: float(v.detach()) for k, v in terms.items()}, 'population_diagnostics': rows,
                        'physical_boundary_exclusion_available': False, 'gradient_guard': guard})+'\n')
            assert loss.isfinite()
            loss.backward()
            assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
            optimizer.step()
            scene.constrain()
            completed = step
            with (directory/'sampling.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': step, 'frame_index': index})+'\n')
            if step % 200 == 0:
                assert all(torch.equal(v, scene.static.state_dict()[k]) for k, v in frozen.items())
                print('UNION_TRAIN', args.phase, step, float(loss.detach()), 'elapsed', round(time.monotonic()-began, 1), flush=True)
                write(UNION_RUNS/'state.json', {'phase': args.phase, 'status': 'smoke' if args.smoke else 'training', 'iteration': step, 'pid': os.getpid()})
            if step % protocol['checkpoint_interval'] == 0:
                save(directory, scene, optimizer, step, stack, args.phase, protocol_hash, parent_hash)
            if not args.smoke and step % protocol['validation_interval'] == 0:
                from .evaluate import evaluate
                evaluate(renderer, scene, args.phase, step, 'val', width, smoke=False)
            if (UNION_RUNS/'stop.request').exists() or args.stop_after and step >= args.stop_after:
                save(directory, scene, optimizer, step, stack, args.phase, protocol_hash, parent_hash)
                write(UNION_RUNS/'state.json', {'phase': args.phase, 'status': 'smoke_saved' if args.smoke else 'saved_stopped', 'iteration': step})
                return
        save(directory, scene, optimizer, completed, stack, args.phase, protocol_hash, parent_hash)
        verify_sources()
        from .evaluate import evaluate
        evaluate(renderer, scene, args.phase, completed, 'val', 1600, smoke=args.smoke)
        # Test stays out of warmup admission/model selection and is read only for final physics.
        if args.phase == 'physics' and not args.smoke:
            evaluate(renderer, scene, args.phase, completed, 'test', 1600, smoke=False)
        write(UNION_RUNS/'state.json', {'phase': args.phase, 'status': 'evaluated_pending_review', 'iteration': completed})
    except Exception as error:
        write(directory/'failure.json', {'error': repr(error), 'last_completed_in_memory': locals().get('completed'),
            'attempted_iteration': locals().get('step'), 'recovery': 'Keep last atomic checkpoint; partial failed step is not a checkpoint'})
        write(UNION_RUNS/'state.json', {'phase': args.phase, 'status': 'failed', 'error': str(error)})
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
