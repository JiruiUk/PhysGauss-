"""Matched I1 continuation with independently switchable ordinal depth."""
import argparse
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..train_static import camera, PIPE
from ..union.runtime import save, restore
from ..motion_integrity.prepare import load_parent, PARENT_SHA
from ..motion_integrity.train import step_loss
from .prepare import EXP, WORK
from .core import geometry_only, ordinal_loss


def extra_loss(scene, renderer, frame, meta, local, plan, cache):
    number = int(Path(frame['file_path']).stem)
    if number not in cache:
        path = WORK/f'pairs/{number:04d}.npz'
        assert sha(path) == plan['depth_receipts'][str(number)]['pairs_sha256']
        with np.load(path) as bank:
            cache[number] = (torch.tensor(bank['pairs'], device='cuda'), torch.tensor(bank['signs'], device='cuda'), camera(frame, meta, plan['depth_width']))
    pairs, signs, cam = cache[number]
    ids = (torch.arange(min(len(pairs), plan['depth_pairs_per_step']), device='cuda')+local*plan['depth_pairs_per_step']) % len(pairs)
    dynamic = geometry_only(scene.dynamic_at(frame['time']))
    package = renderer(cam, dynamic, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
    return ordinal_loss(package['depth'], package['alpha'], pairs[ids], signs[ids], plan['ordinal_sharpness'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--enable-ordinal-depth', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stop-after', type=int)
    args = p.parse_args()
    if not args.enable_ordinal_depth:
        raise SystemExit('Ordinal experiment is opt-in')
    torch.set_num_threads(4)
    plan, protocol_hash = read(EXP/'protocol.json'), sha(EXP/'protocol.json')
    if not args.smoke:
        assert read(EXP/'preflight.json')['passed']
    directory = WORK/('_smoke' if args.smoke else 'formal')
    directory.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = open(WORK/'training.lock', 'a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit('Ordinal-depth training already running')
    try:
        renderer, scene, opt, parent, base = load_parent()
        if (directory/'checkpoint.pt').exists():
            state = restore(directory/'checkpoint.pt', scene, opt, protocol_hash)
            assert state['phase'] == plan['arm'] and state['parent_sha256'] == PARENT_SHA
            start, stack = state['iteration'], list(state['camera_stack'])
            write(directory/f'resume_{start:04d}.json', {'iteration': start, 'full_state_restored': True})
        else:
            start, stack = 0, list(parent['camera_stack'])
            save(directory, scene, opt, 0, stack, plan['arm'], protocol_hash, PARENT_SHA)
            shutil.copyfile(directory/'checkpoint.pt', directory/'initial.pt')
        assert sha(ROOT/plan['graph']) == plan['graph_sha256']
        with np.load(ROOT/plan['graph']) as data:
            graph = tuple(torch.as_tensor(data[k], device='cuda') for k in ['edges','rest','weight'])
        meta = read(DATA/'transforms_train.json')
        for i, frame in enumerate(meta['frames']):
            frame['_index'] = i
        cache, depth_cache, began = {}, {}, time.monotonic()
        frozen = {k: v.detach().clone() for k, v in scene.static.state_dict().items()}
        target = plan['smoke_updates'] if args.smoke else plan['actual_updates']
        for local in range(start+1, target+1):
            if (WORK/'stop.request').exists():
                save(directory, scene, opt, local-1, stack, plan['arm'], protocol_hash, PARENT_SHA)
                return
            if not stack:
                stack = random.sample(range(120), 120)
            index = stack.pop()
            frame = meta['frames'][index]
            loss, record = step_loss(scene, renderer, opt, frame, meta, cache,
                                     6000+local, local, True, plan, base, graph)
            depth = extra_loss(scene, renderer, frame, meta, local, plan, depth_cache)
            weight = plan['ordinal_weight']*min(1., local/plan['ordinal_ramp'])
            record.update(ordinal=float(depth.detach()), ordinal_weight=weight)
            if local == 1:
                named = [(n, v) for n, v in scene.named_parameters() if v.requires_grad]
                gs = torch.autograd.grad(depth, [v for _, v in named], allow_unused=True, retain_graph=True)
                record['depth_gradient_norm'] = {n: 0. if g is None else float(g.norm()) for (n, _), g in zip(named, gs)}
                for prefix in ['pools.0._xyz', 'grids.', 'networks.']:
                    assert sum(v for n, v in record['depth_gradient_norm'].items() if n.startswith(prefix)) > 0
                assert all(v == 0 for n, v in record['depth_gradient_norm'].items() if n.startswith('physics.') or any(x in n for x in ['_features','_opacity','_scaling','_rotation']))
            loss = loss+weight*depth
            assert loss.isfinite()
            loss.backward()
            assert all(v.grad is None or v.grad.isfinite().all() for v in scene.parameters())
            opt.step(); scene.constrain()
            with (directory/'steps.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            if local in [1,40,80,100] or local % 200 == 0:
                assert all(torch.equal(v, scene.static.state_dict()[k]) for k, v in frozen.items())
                print('ORDINAL_TRAIN', local, 'loss', float(loss.detach()), 'ordinal', record['ordinal'], 'elapsed', round(time.monotonic()-began,1), flush=True)
                write(EXP/'progress.json', {'mode': 'smoke' if args.smoke else 'formal', 'actual_updates': local, 'target': target, 'acceptance':'pending'})
            if local in [40,80,100] or local % 500 == 0 or args.stop_after and local >= args.stop_after:
                save(directory, scene, opt, local, stack, plan['arm'], protocol_hash, PARENT_SHA)
            if args.stop_after and local >= args.stop_after:
                return
        save(directory, scene, opt, target, stack, plan['arm'], protocol_hash, PARENT_SHA)
    except Exception as error:
        write(directory/'failure.json', {'error': repr(error), 'attempted_update': locals().get('local')})
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
