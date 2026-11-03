"""Matched continuation of the unchanged S3-U field, with an optional prior."""
import argparse
import json
import os
import random
import shutil
import time
import numpy as np
import torch
from ..prepare_data import DATA, ROOT, read, write, sha
from ..train_static import camera
from ..stage2.train import image
from ..union.train import reconstruction
from ..union.runtime import restore, save
from .prepare import EXP, WORK, PARENT_SHA, load_parent
from .core import strain_loss


def step_loss(scene, renderer, optimizer, frame, meta, cache, step, local, enabled, plan, base, graph):
    index = int(frame['_index'])
    if index not in cache:
        rgba = image(frame, plan['width'])
        cache[index] = (rgba, torch.tensor(rgba[..., :3]*rgba[..., 3:4]).permute(2, 0, 1), camera(frame, meta, plan['width']))
    rgba, cpu_gt, cam = cache[index]
    gt = cpu_gt.cuda()
    optimizer.zero_grad(set_to_none=True)
    package = scene.render(renderer, cam, frame['time'])
    rgb = reconstruction(package, gt, rgba, scene, cam)
    loss = rgb
    record = {'absolute_step': step, 'local_step': local, 'frame_index': index, 'frame': frame['file_path'], 'rgb': float(rgb.detach())}
    if step % base['physics_every'] == 0:
        from variants.momentum_guard.balance import balance
        terms, rows = scene.physics.loss(scene, frame['time'], step, base['physics_samples_global'], base['time_step'])
        physical = base['kinematic_weight']*terms['kinematic']+base['CMR_weight']*terms['CMR']
        groups = {'grid': list(scene.grids.parameters()), 'deformation': list(scene.networks.parameters()), 'canonical': [scene.pools[0]._xyz]}
        guarded, guard = balance(rgb, physical, groups, list(scene.physics.parameters()), base['shared_gradient_cap'], 1.)
        loss = loss+guarded
        record['physics'] = {'terms': {k: float(v.detach()) for k, v in terms.items()}, 'diagnostics': rows, 'guard': guard}
    if enabled:
        edges, rest, weights = graph
        ids = (torch.arange(min(plan['edges_per_update'], len(edges)), device=edges.device)+local*plan['edges_per_update']) % len(edges)
        edge, length, weight = edges[ids], rest[ids], weights[ids]
        extra_index = (local*37) % len(meta['frames'])
        extra_time = meta['frames'][extra_index]['time']
        observed_positions = package['means3D_deform'][len(scene.static.get_xyz):]
        prior = .5*(strain_loss(observed_positions, edge, length, weight)+
                    strain_loss(scene.dynamic_at(extra_time).get_xyz, edge, length, weight))
        if local == 1:
            named = [(n, p) for n, p in scene.named_parameters() if p.requires_grad]
            gradients = torch.autograd.grad(prior, [p for _, p in named], allow_unused=True, retain_graph=True)
            record['prior_gradient_norm'] = {n: 0. if g is None else float(g.detach().norm()) for (n, _), g in zip(named, gradients)}
            for prefix in ['pools.0._xyz', 'grids.', 'networks.']:
                assert sum(v for n, v in record['prior_gradient_norm'].items() if n.startswith(prefix)) > 0, prefix
            assert all(v == 0 for n, v in record['prior_gradient_norm'].items() if n.startswith('physics.') or '_opacity' in n or '_features' in n)
        multiplier = plan['strain_weight']*min(1., local/plan['ramp_updates'])
        loss = loss+multiplier*prior
        record.update(prior=float(prior.detach()), prior_weight=multiplier, prior_time_index=extra_index, edge_offset=int(ids[0]))
    assert loss.isfinite()
    return loss, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-motion-integrity', action='store_true')
    parser.add_argument('--arm', choices=['I0_control', 'I1_curve_isometry'], default='I0_control')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if not args.enable_motion_integrity:
        raise SystemExit('Motion integrity is opt-in; pass --enable-motion-integrity')
    torch.set_num_threads(4)
    plan = read(EXP/'protocol.json')
    protocol_hash = sha(EXP/'protocol.json')
    if not args.smoke:
        assert read(EXP/'preflight.json')['passed'], 'Preflight must pass before formal training'
    assert sha(ROOT/plan['graph']) == plan['graph_sha256']
    enabled = args.arm == 'I1_curve_isometry'
    directory = WORK/('_smoke' if args.smoke else 'formal')/args.arm
    directory.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = open(WORK/'training.lock', 'a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit('Motion-integrity training already running')
    try:
        renderer, scene, optimizer, initial, base = load_parent()
        if (directory/'checkpoint.pt').exists():
            state = restore(directory/'checkpoint.pt', scene, optimizer, protocol_hash)
            assert state['phase'] == args.arm and state['parent_sha256'] == PARENT_SHA
            start, stack = state['iteration'], list(state['camera_stack'])
        else:
            start, stack = 0, list(initial['camera_stack'])
            save(directory, scene, optimizer, 0, stack, args.arm, protocol_hash, PARENT_SHA)
            shutil.copyfile(directory/'checkpoint.pt', directory/'initial.pt')
        frozen = {k: v.detach().clone() for k, v in scene.static.state_dict().items()}
        with np.load(ROOT/plan['graph']) as value:
            graph = tuple(torch.as_tensor(value[k], device='cuda') for k in ['edges', 'rest', 'weight'])
        meta = read(DATA/'transforms_train.json')
        for i, frame in enumerate(meta['frames']):
            frame['_index'] = i
        updates = 100 if args.smoke else plan['actual_updates']
        cache, began = {}, time.monotonic()
        for local in range(start+1, updates+1):
            if (WORK/'stop.request').exists():
                save(directory, scene, optimizer, local-1, stack, args.arm, protocol_hash, PARENT_SHA)
                return
            if not stack:
                stack = random.sample(range(len(meta['frames'])), len(meta['frames']))
            index = stack.pop()
            loss, record = step_loss(scene, renderer, optimizer, meta['frames'][index], meta, cache,
                                     6000+local, local, enabled, plan, base, graph)
            loss.backward()
            assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
            optimizer.step()
            scene.constrain()
            with (directory/'steps.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            if local in [1, 40, 80, 100] or local % 200 == 0:
                assert all(torch.equal(v, scene.static.state_dict()[k]) for k, v in frozen.items())
                print('MOTION_TRAIN', args.arm, local, float(loss.detach()), record.get('prior'), 'elapsed', round(time.monotonic()-began, 1), flush=True)
                write(EXP/'progress.json', {'arm': args.arm, 'mode': 'smoke' if args.smoke else 'formal',
                      'actual_updates': local, 'target': updates, 'pid': os.getpid(), 'acceptance': 'pending'})
            if local in [40, 80, 100] or local % 500 == 0 or args.stop_after and local >= args.stop_after:
                save(directory, scene, optimizer, local, stack, args.arm, protocol_hash, PARENT_SHA)
                if args.smoke and local in [40, 80, 100]:
                    shutil.copyfile(directory/'checkpoint.pt', directory/f'step{local:04d}.pt')
            if args.stop_after and local >= args.stop_after:
                return
        save(directory, scene, optimizer, updates, stack, args.arm, protocol_hash, PARENT_SHA)
        assert sha(ROOT/plan['parent']) == PARENT_SHA
    except Exception as error:
        write(directory/'failure.json', {'error': repr(error), 'attempted_local_update': locals().get('local')})
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
