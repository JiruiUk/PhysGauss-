"""Native sparse-motion continuation with exactly inherited optimization state."""
import argparse
import json
import random
import shutil
import time
import numpy as np
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..union.runtime import save, restore
from ..motion_integrity.prepare import load_parent, PARENT_SHA
from ..motion_integrity.train import step_loss
from .prepare import EXP, WORK
from .core import SparseDeform


def install(scene, plan):
    path = ROOT/plan['controls']
    assert sha(path) == plan['controls_sha256']
    with np.load(path) as data:
        adapter = SparseDeform(scene.deforms[0], torch.tensor(data['xyz'], device='cuda'),
                               torch.tensor(data['radius'], device='cuda'), mix=0)
    scene.deforms[0] = adapter
    return adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--enable-sparse-motion', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stop-after', type=int)
    args = p.parse_args()
    if not args.enable_sparse_motion:
        raise SystemExit('Sparse motion is opt-in; pass --enable-sparse-motion')
    torch.set_num_threads(4)
    plan = read(EXP/'protocol.json')
    plan_hash = sha(EXP/'protocol.json')
    if not args.smoke:
        preflight = read(EXP/'preflight.json')
        assert preflight['passed'] and preflight['actual_optimizer_updates'] >= 500
        assert preflight['fully_sparse_optimizer_updates'] >= 100
    directory = WORK/('_smoke' if args.smoke else 'formal')
    directory.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = open(WORK/'training.lock', 'a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit('Sparse-motion training already running')
    try:
        renderer, scene, opt, parent, base = load_parent()
        adapter = install(scene, plan)
        if (directory/'checkpoint.pt').exists():
            state = restore(directory/'checkpoint.pt', scene, opt, plan_hash)
            assert state['phase'] == plan['arm'] and state['parent_sha256'] == PARENT_SHA
            start, stack = state['iteration'], list(state['camera_stack'])
            adapter.mix = min(1., start/plan['motion_mix_ramp_updates'])
            write(directory/f'resume_{start:04d}.json', {'iteration': start, 'recovered_mix': adapter.mix,
                  'controls_sha256': plan['controls_sha256'], 'model_Adam_RNG_camera_and_graph_restored': True})
        else:
            start, stack = 0, list(parent['camera_stack'])
            save(directory, scene, opt, 0, stack, plan['arm'], plan_hash, PARENT_SHA)
            shutil.copyfile(directory/'checkpoint.pt', directory/'initial.pt')
        frozen = {k: v.detach().clone() for k, v in scene.static.state_dict().items()}
        assert sha(ROOT/plan['graph']) == plan['graph_sha256']
        with np.load(ROOT/plan['graph']) as value:
            graph = tuple(torch.as_tensor(value[k], device='cuda') for k in ['edges', 'rest', 'weight'])
        meta = read(DATA/'transforms_train.json')
        for i, frame in enumerate(meta['frames']):
            frame['_index'] = i
        cache, began = {}, time.monotonic()
        # Exercise the full400-step transition AND100 fully sparse updates before
        # admitting formal training; a100-step partial-mix smoke is insufficient.
        target = 500 if args.smoke else plan['actual_updates']
        for local in range(start+1, target+1):
            if (WORK/'stop.request').exists():
                save(directory, scene, opt, local-1, stack, plan['arm'], plan_hash, PARENT_SHA)
                return
            if not stack:
                stack = random.sample(range(120), 120)
            index = stack.pop()
            adapter.mix = min(1., local/plan['motion_mix_ramp_updates'])
            loss, record = step_loss(scene, renderer, opt, meta['frames'][index], meta, cache,
                                     6000+local, local, True, plan, base, graph)
            record['sparse_mix'] = adapter.mix
            loss.backward()
            assert all(param.grad is None or param.grad.isfinite().all() for param in scene.parameters())
            opt.step()
            scene.constrain()
            with (directory/'steps.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            if local in [1, 40, 80, 100] or local % 200 == 0:
                assert all(torch.equal(v, scene.static.state_dict()[k]) for k, v in frozen.items())
                print('SPARSE_TRAIN', local, 'mix', adapter.mix, 'loss', float(loss.detach()), 'prior', record['prior'], 'elapsed', round(time.monotonic()-began, 1), flush=True)
                write(EXP/'progress.json', {'mode': 'smoke' if args.smoke else 'formal', 'actual_updates': local,
                      'target': target, 'sparse_mix': adapter.mix, 'integrity_acceptance': 'pending'})
            if local in [40, 80, 100] or local % 500 == 0 or args.stop_after and local >= args.stop_after:
                save(directory, scene, opt, local, stack, plan['arm'], plan_hash, PARENT_SHA)
                if args.smoke and local in [40, 80, 100]:
                    shutil.copyfile(directory/'checkpoint.pt', directory/f'step{local:04d}.pt')
            if args.stop_after and local >= args.stop_after:
                return
        save(directory, scene, opt, target, stack, plan['arm'], plan_hash, PARENT_SHA)
    except Exception as error:
        write(directory/'failure.json', {'error': repr(error), 'attempted_update': locals().get('local')})
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
