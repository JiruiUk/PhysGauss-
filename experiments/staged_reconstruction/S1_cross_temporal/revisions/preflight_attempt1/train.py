"""Matched fixed/progressive confidence arms using unchanged native PIDG renderer."""
import argparse
import json
import os
import random
import msvcrt
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as functional
from ..prepare_data import ROOT, DATA, read, write, sha
from ..static_gaussian import StaticGaussian
from ..train_static import setup, camera, render_static, optimizer_for
from ..automatic_static.discover import rgb
from .prepare import REPORT, RUNS, verify
from .core import confidence_step, weighted_objective, summary


def confidence_hash(values):
    import hashlib
    return hashlib.sha256(b''.join(n.encode()+np.asarray(values[n], dtype=np.float32).tobytes()
                                  for n in sorted(values))).hexdigest()


def checkpoint(directory, model, opt, step, stack, confidence, history, arm, protocol_hash):
    state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'iteration': step,
        'camera_stack': stack, 'confidence': confidence, 'confidence_history': history,
        'confidence_sha256': confidence_hash(confidence), 'arm': arm,
        'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
        'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
        'protocol_sha256': protocol_hash, 'source_manifest_sha256': sha(REPORT/'source_freeze.json'),
        'static_only': True, 'dynamic_parameters': 0}
    temporary = directory/'checkpoint.tmp'; path = directory/'checkpoint.pt'
    torch.save(state, temporary); os.replace(temporary, path)
    write(directory/'checkpoint.json', {'iteration': step, 'sha256': sha(path),
                                      'confidence_sha256': state['confidence_sha256']})


def restore(directory, model, opt, arm, protocol_hash):
    path = directory/'checkpoint.pt'
    assert sha(path) == read(directory/'checkpoint.json')['sha256']
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['arm'] == arm and state['protocol_sha256'] == protocol_hash
    assert state['source_manifest_sha256'] == sha(REPORT/'source_freeze.json')
    assert state['confidence_sha256'] == confidence_hash(state['confidence'])
    model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
    random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch']); torch.cuda.set_rng_state_all(state['rng_cuda'])
    return state['iteration'], state['camera_stack'], state['confidence'], state['confidence_history']


@torch.no_grad()
def refresh(renderer, model, frames, meta, confidence, geometry, targets, protocol, step):
    before = summary(confidence); added, revoked = 0, 0
    for frame in frames:
        n = str(int(frame['file_path'].replace('\\', '/').split('/')[-1]))
        package = render_static(renderer, camera(frame, meta, 400), model)
        pred = package['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        alpha = package['alpha'].squeeze().cpu().numpy()
        residual = cv2.blur(np.mean(np.abs(pred-targets[n]), 2).astype(np.float32), (5, 5))
        photo = np.exp(-residual/.1)
        old = confidence[n]; updated = confidence_step(old, geometry[n], photo, alpha > .3, protocol['confidence_rate'])
        added += int(((old < .5)&(updated >= .5)).sum())
        revoked += int(((old >= .5)&(updated < .5)).sum())
        confidence[n] = updated
    return {'step': step, 'before': before['positive_pixels'], **summary(confidence),
            'newly_admitted': added, 'revoked': revoked, 'confidence_sha256': confidence_hash(confidence)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-cross-temporal-static', action='store_true')
    parser.add_argument('--arm', choices=['fixed', 'progressive'], default='fixed')
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if not args.enable_cross_temporal_static:
        raise SystemExit('Cross-temporal reconstruction is opt-in')
    verify()
    assert read(REPORT/'candidate_review.json')['accepted_for_bounded_pilot']
    assert read(REPORT/'test_receipt.json')['passed']
    if not args.smoke: assert read(REPORT/'smoke_verification.json')['passed']
    assert not (RUNS/'stop.request').exists(), 'Saved user stop request; do not resume'
    lock = (RUNS/'training.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    directory = RUNS/('smoke' if args.smoke else 'formal')/args.arm
    directory.mkdir(parents=True, exist_ok=True)
    try:
        protocol = read(REPORT/'protocol.json'); protocol_hash = sha(REPORT/'protocol.json')
        torch.set_num_threads(4); cv2.setNumThreads(2)
        random.seed(protocol['seed']); np.random.seed(protocol['seed']); torch.manual_seed(protocol['seed'])
        renderer = setup(); model = StaticGaussian(RUNS/'initialization.npz')
        with torch.no_grad(): model._opacity.fill_(float(np.log(.1/.9)))
        opt = optimizer_for(model)
        meta = read(DATA/'transforms_train.json')
        frames = sorted(meta['frames'], key=lambda f: int(f['file_path'].replace('\\', '/').split('/')[-1]))
        with np.load(RUNS/'initial_confidence.npz') as data: confidence = {n: data[n].copy() for n in data.files}
        with np.load(RUNS/'geometric_confidence.npz') as data: geometry = {n: data[n].copy() for n in data.files}
        pool = [f for f in frames if (geometry[str(int(f['file_path'].replace('\\', '/').split('/')[-1]))] >= .5).sum() >= 64]
        assert pool
        write(directory/'sampling_contract.json', {'frames': [f['file_path'] for f in pool], 'seed': protocol['seed'],
            'pool_is_common_geometric_candidate_frames': True, 'zero_supervision_views_are_logged_not_background': True})
        if not (directory/'checkpoint.pt').exists():
            checkpoint(directory, model, opt, 0, [], confidence, [], args.arm, protocol_hash)
            (directory/'initial.pt').write_bytes((directory/'checkpoint.pt').read_bytes())
        step0, stack, confidence, history = restore(directory, model, opt, args.arm, protocol_hash)
        # Read-only native adapter parity must preserve restored RNG in full.
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
        from scene_PIDG.gaussian_model import GaussianModel
        native = GaussianModel(3); native.active_sh_degree = 3
        for name, parameter in model.named_parameters(): setattr(native, name, parameter)
        with torch.no_grad():
            cam = camera(pool[0], meta, 800)
            a = render_static(renderer, cam, model); b = render_static(renderer, cam, native)
            assert torch.equal(a['render'], b['render']) and torch.equal(a['alpha'], b['alpha'])
        del a, b, native
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
        write(REPORT/'native_parity.json', {'passed': True, 'native_original_GaussianModel_exact': True})
        target_steps = 80 if args.smoke else protocol['stage_updates']
        interval = 20 if args.smoke else protocol['confidence_interval']
        small_targets = {str(int(f['file_path'].replace('\\', '/').split('/')[-1])): rgb(f, 400) for f in frames}
        cache, cameras, mask_cache = {}, {}, {}; width = None; began = time.monotonic()
        write(RUNS/'state.json', {'status': 'training', 'arm': args.arm, 'iteration': step0, 'pid': os.getpid(), 'smoke': args.smoke})
        for step in range(step0+1, target_steps+1):
            new_width = 800 if step <= 8000 else 1600
            if width != new_width:
                cache.clear(); cameras.clear(); mask_cache.clear(); width = new_width
            if not stack: stack = random.sample(range(len(pool)), len(pool))
            index = stack.pop(); frame = pool[index]
            n = str(int(frame['file_path'].replace('\\', '/').split('/')[-1]))
            if index not in cache:
                cache[index] = torch.from_numpy(rgb(frame, width)).permute(2, 0, 1)
                cameras[index] = camera(frame, meta, width)
            if index not in mask_cache:
                mask_cache[index] = functional.interpolate(torch.from_numpy(confidence[n])[None, None].cuda(),
                    size=(width*9//16, width), mode='nearest')[0, 0]
            gt = cache[index].cuda(); weight = mask_cache[index]
            opt.zero_grad(set_to_none=True)
            package = render_static(renderer, cameras[index], model)
            loss = weighted_objective(package['render'], package['alpha'], gt, weight, protocol['admission_threshold'])
            loss = loss+.01*(model._xyz-model.initial_xyz).square().mean()
            assert torch.isfinite(loss); loss.backward()
            assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
            opt.step(); model.constrain()
            with (directory/'sampling.jsonl').open('a') as log:
                log.write(json.dumps({'step': step, 'frame_index': index, 'frame': n,
                    'supervised_pixels': int((weight >= .5).sum())})+'\n')
            if step % 20 == 0:
                with (directory/'losses.jsonl').open('a') as log:
                    log.write(json.dumps({'step': step, 'loss': float(loss.detach()), 'width': width})+'\n')
            if args.arm == 'progressive' and step % interval == 0:
                event = refresh(renderer, model, frames, meta, confidence, geometry, small_targets, protocol, step)
                history.append(event); mask_cache.clear()
                write(directory/'confidence_history.json', history)
                print('CROSS_TEMPORAL_REFRESH', args.arm, step, event['positive_pixels'], event['newly_admitted'], event['revoked'], flush=True)
            if step % 200 == 0:
                print('CROSS_TEMPORAL_TRAIN', args.arm, step, float(loss.detach()), 'elapsed', round(time.monotonic()-began, 1), flush=True)
                write(RUNS/'state.json', {'status': 'training', 'arm': args.arm, 'iteration': step, 'pid': os.getpid(), 'smoke': args.smoke})
            if step % 1000 == 0: checkpoint(directory, model, opt, step, stack, confidence, history, args.arm, protocol_hash)
            if (RUNS/'stop.request').exists() or args.stop_after and step >= args.stop_after:
                checkpoint(directory, model, opt, step, stack, confidence, history, args.arm, protocol_hash)
                write(RUNS/'state.json', {'status': 'saved_stopped', 'arm': args.arm, 'iteration': step, 'smoke': args.smoke})
                return
        checkpoint(directory, model, opt, target_steps, stack, confidence, history, args.arm, protocol_hash)
        verify()
        write(RUNS/'state.json', {'status': 'trained_pending_evaluation', 'arm': args.arm, 'iteration': target_steps, 'smoke': args.smoke})
    except Exception as error:
        write(RUNS/'state.json', {'status': 'failed', 'arm': args.arm, 'error': str(error), 'pid': os.getpid()})
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__': main()
