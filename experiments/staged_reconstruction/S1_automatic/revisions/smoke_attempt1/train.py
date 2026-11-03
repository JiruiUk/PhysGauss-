"""Independent static 3DGS training from image-geometric masks, no semantic labels."""
import argparse
import json
import os
from pathlib import Path
import random
import msvcrt
import numpy as np
from PIL import Image
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..static_gaussian import StaticGaussian
from ..train_static import setup, camera, render_static, optimizer_for, save, restore
from .discover import REPORT, RUNS, rgb
from .geometry import static_objective
from .initialize import verify


def load_frame(frame, width, masks):
    target = torch.tensor(rgb(frame, width)).permute(2, 0, 1).cuda()
    number = str(int(Path(frame['file_path']).stem))
    positive = np.asarray(Image.fromarray(masks[number]).resize((width, width*9//16), Image.Resampling.NEAREST))
    return target, torch.tensor(positive, device='cuda')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-automatic-static', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if not args.enable_automatic_static:
        raise SystemExit('Automatic static 3DGS training is opt-in')
    verify()
    assert read(REPORT/'mask_review.json')['accepted_for_conditional_training'], 'Actual mask review required'
    if not args.smoke:
        assert read(REPORT/'smoke_verification.json')['passed']
    if (RUNS/'stop.request').exists():
        raise SystemExit('Saved stop request present; do not resume paused training')
    lock = (RUNS/'training.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        protocol = read(REPORT/'protocol.json')
        protocol_hash = sha(REPORT/'protocol.json')
        torch.set_num_threads(4)
        random.seed(protocol['seed']); np.random.seed(protocol['seed']); torch.manual_seed(protocol['seed'])
        renderer = setup()
        model = StaticGaussian(RUNS/'initialization.npz')
        opt = optimizer_for(model)
        directory = RUNS/('smoke' if args.smoke else 'formal')
        directory.mkdir(parents=True, exist_ok=True)
        meta = read(DATA/'transforms_train.json')
        masks = np.load(RUNS/'inferred_masks.npz')
        frames = [f for f in meta['frames'] if masks[str(int(Path(f['file_path']).stem))].sum() >= 64]
        assert frames, 'No image-geometric supervision'
        write(directory/'sampling_contract.json', {'frames': [f['file_path'] for f in frames], 'seed': protocol['seed']})
        if not (directory/'initial.pt').exists():
            save(directory, model, opt, 0, [], protocol_hash)
            (directory/'initial.pt').write_bytes((directory/'checkpoint.pt').read_bytes())
        step0, stack = restore(directory, model, opt, protocol_hash)
        target_steps = protocol['stage_updates']
        cache, cameras = {}, {}
        width = None
        # Exact native Gaussian adapter parity; this never loads a historical model.
        frame = frames[0]
        from scene_PIDG.gaussian_model import GaussianModel
        native = GaussianModel(3); native.active_sh_degree = 3
        for name, value in model.named_parameters():
            setattr(native, name, value)
        with torch.no_grad():
            cam = camera(frame, meta, 1600)
            a = render_static(renderer, cam, model); b = render_static(renderer, cam, native)
            assert torch.equal(a['render'], b['render']) and torch.equal(a['alpha'], b['alpha'])
        write(REPORT/'native_parity.json', {'passed': True, 'exact_original_GaussianModel_parity': True})
        write(RUNS/'state.json', {'status': 'training', 'pid': os.getpid(), 'iteration': step0, 'smoke': args.smoke})
        for step in range(step0+1, target_steps+1):
            new_width = 800 if step <= 8000 else 1600
            if width != new_width:
                cache.clear(); cameras.clear(); width = new_width
            if not stack:
                stack = random.sample(range(len(frames)), len(frames))
            index = stack.pop()
            frame = frames[index]
            if index not in cache:
                cache[index] = load_frame(frame, width, masks)
                cameras[index] = camera(frame, meta, width)
            gt, mask = cache[index]
            opt.zero_grad(set_to_none=True)
            package = render_static(renderer, cameras[index], model)
            loss = static_objective(package['render'], package['alpha'], gt, mask)
            loss = loss+.01*(model._xyz-model.initial_xyz).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
            opt.step(); model.constrain()
            with (directory/'sampling.jsonl').open('a') as log:
                log.write(json.dumps({'step': step, 'frame_index': index})+'\n')
            if step % 20 == 0:
                with (directory/'losses.jsonl').open('a') as log:
                    log.write(json.dumps({'step': step, 'loss': float(loss.detach()), 'width': width})+'\n')
            if step % 200 == 0:
                print('AUTOMATIC_STATIC_TRAIN', step, float(loss.detach()), flush=True)
                write(RUNS/'state.json', {'status': 'training', 'iteration': step, 'pid': os.getpid(), 'smoke': args.smoke})
            if step % 1000 == 0:
                save(directory, model, opt, step, stack, protocol_hash)
            if (RUNS/'stop.request').exists() or args.stop_after and step >= args.stop_after:
                save(directory, model, opt, step, stack, protocol_hash)
                write(RUNS/'state.json', {'status': 'saved_stopped', 'iteration': step, 'smoke': args.smoke})
                return
        save(directory, model, opt, target_steps, stack, protocol_hash)
        verify()
        write(RUNS/'state.json', {'status': 'trained_pending_evaluation', 'iteration': target_steps, 'smoke': args.smoke})
        print('AUTOMATIC_STATIC_TRAINING_COMPLETE', target_steps, flush=True)
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
