"""Original field/physics continuation with one optional RGB reference change."""
import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
import torch
from ..prepare_data import DATA, read, write, sha
from ..train_static import camera, PIPE
from ..stage2.train import image
from ..union.runtime import restore, save
from .prepare import EXP, WORK, PARENT, PARENT_SHA, load_parent, verify
from .core import reconstruction, conditional_static_mask


def step_loss(scene, renderer, optimizer, frame, meta, cache, local, enabled, plan, base):
    index = int(frame['_index'])
    if index not in cache:
        rgba = image(frame, plan['width'])
        cam = camera(frame, meta, plan['width'])
        static = None
        if enabled:
            with torch.no_grad():
                static = renderer(cam, scene.static, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)['render'].cpu()
        cache[index] = (rgba, torch.tensor(rgba[..., :3]*rgba[..., 3:4]).permute(2, 0, 1), cam, static)
    rgba, cpu_gt, cam, static = cache[index]
    gt = cpu_gt.cuda(); optimizer.zero_grad(set_to_none=True)
    package = scene.render(renderer, cam, frame['time'])
    static_gpu = static.cuda() if enabled else None
    rgb = reconstruction(package, gt, rgba, scene, cam, enabled=enabled, static_render=static_gpu)
    loss = rgb; step = 6000+local
    record = {'absolute_step': step, 'local_step': local, 'frame_index': index,
        'frame': frame['file_path'], 'rgb': float(rgb.detach()), 'neutralization_enabled': enabled}
    if enabled:
        mask = conditional_static_mask(gt)
        record['static_reference_RGB_MAE'] = float((static_gpu[:, mask]-gt[:, mask]).abs().mean()) if mask.any() else 0.
    if step % base['physics_every'] == 0:
        from variants.momentum_guard.balance import balance
        terms, rows = scene.physics.loss(scene, frame['time'], step, base['physics_samples_global'], base['time_step'])
        physical = base['kinematic_weight']*terms['kinematic']+base['CMR_weight']*terms['CMR']
        groups = {'grid': list(scene.grids.parameters()), 'deformation': list(scene.networks.parameters()), 'canonical': [scene.pools[0]._xyz]}
        guarded, guard = balance(rgb, physical, groups, list(scene.physics.parameters()), base['shared_gradient_cap'], 1.)
        loss = loss+guarded
        record['physics'] = {'terms': {k: float(v.detach()) for k, v in terms.items()}, 'diagnostics': rows, 'guard': guard}
    assert loss.isfinite()
    return loss, record


def main():
    p = argparse.ArgumentParser(); p.add_argument('--enable-background-residual', action='store_true')
    p.add_argument('--arm', choices=['I10_control', 'I10_static_residual'], default='I10_control')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--stop-after', type=int); args = p.parse_args()
    if not args.enable_background_residual:
        raise SystemExit('Background residual training is opt-in')
    torch.set_num_threads(4); plan = verify(); protocol_hash = sha(EXP/'protocol.json')
    if not args.smoke:
        assert read(EXP/'preflight.json')['passed'], 'Formal training requires actual passed preflight'
    enabled = args.arm == 'I10_static_residual'
    directory = WORK/('_smoke' if args.smoke else 'formal')/args.arm; directory.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = open(WORK/'training.lock', 'a+b'); lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit('Background residual training already running')
    try:
        assert not (WORK/'stop.request').exists(), 'Explicit stop request is present'
        renderer, scene, opt, initial, base = load_parent()
        if (directory/'checkpoint.pt').exists():
            state = restore(directory/'checkpoint.pt', scene, opt, protocol_hash)
            assert state['phase'] == args.arm and state['parent_sha256'] == PARENT_SHA
            start, stack = state['iteration'], list(state['camera_stack'])
        else:
            start, stack = 0, list(initial['camera_stack'])
            save(directory, scene, opt, 0, stack, args.arm, protocol_hash, PARENT_SHA)
            shutil.copyfile(directory/'checkpoint.pt', directory/'initial.pt')
        with (directory/'launches.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'start': start, 'checkpoint_sha256': sha(directory/'checkpoint.pt'),
                                     'utc': datetime.now(timezone.utc).isoformat()})+'\n')
        frozen = {k: v.detach().clone() for k, v in scene.static.state_dict().items()}
        meta = read(DATA/'transforms_train.json')
        for i, frame in enumerate(meta['frames']):
            frame['_index'] = i
        updates = plan['smoke_updates'] if args.smoke else plan['actual_updates']
        cache = {}; began = time.monotonic(); completed = start
        for local in range(start+1, updates+1):
            if (WORK/'stop.request').exists():
                save(directory, scene, opt, completed, stack, args.arm, protocol_hash, PARENT_SHA); return
            if not stack:
                stack = random.sample(range(len(meta['frames'])), len(meta['frames']))
            index = stack.pop()
            loss, record = step_loss(scene, renderer, opt, meta['frames'][index], meta, cache, local, enabled, plan, base)
            loss.backward()
            assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
            opt.step(); scene.constrain(); completed = local
            with (directory/'steps.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            if local in [1, 40, 80, 100] or local % 200 == 0:
                assert all(torch.equal(v, scene.static.state_dict()[k]) for k, v in frozen.items())
                print('BACKGROUND_RESIDUAL_TRAIN', args.arm, local, float(loss.detach()), 'elapsed', round(time.monotonic()-began, 1), flush=True)
                write(EXP/'progress.json', {'arm': args.arm, 'mode': 'smoke' if args.smoke else 'formal',
                    'actual_updates': local, 'target': updates, 'pid': os.getpid(), 'acceptance': 'pending'})
            if local in [40, 80, 100] or local % 500 == 0 or args.stop_after and local >= args.stop_after:
                save(directory, scene, opt, local, stack, args.arm, protocol_hash, PARENT_SHA)
                if args.smoke and local in [40, 80, 100]:
                    shutil.copyfile(directory/'checkpoint.pt', directory/f'step{local:04d}.pt')
            if args.stop_after and local >= args.stop_after:
                return
        save(directory, scene, opt, completed, stack, args.arm, protocol_hash, PARENT_SHA); verify()
        write(directory/'terminal.json', {'status': 'completed', 'iteration': completed, 'sha256': sha(directory/'checkpoint.pt')})
    except Exception as error:
        write(directory/'failure.json', {'error': repr(error), 'attempted_local_update': locals().get('local'),
            'last_completed_in_memory': locals().get('completed'), 'last_atomic_checkpoint_kept': True})
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
