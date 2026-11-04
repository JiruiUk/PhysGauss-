"""Evaluate actual GT and unchanged fixed new views; reference is training-only."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..train_static import camera, PIPE
from ..stage2.train import image
from ..model import observation_masks
from ..union.core import dynamic_mask
from ..motion_integrity.evaluate import pixels, tile, fingerprint
from variants.view_completeness.metrics import shape_proxies, summarize
from .prepare import EXP, WORK, PARENT, PARENT_SHA, load_parent, verify

OUT = ROOT/'output/staged_reconstruction/background_residual'


@torch.no_grad()
def validation(renderer, scene, state):
    scene.load_state_dict(state['model'])
    meta = read(ROOT/'data/derived/kinematic_resplit/transforms_val.json'); rows = []
    for frame in meta['frames']:
        rgba = image(frame, 1600)
        gt = torch.tensor(rgba[..., :3]*rgba[..., 3:4], device='cuda').permute(2, 0, 1)
        cam = camera(frame, meta, 1600)
        pred = scene.render(renderer, cam, frame['time'])['render'].clamp(0, 1)
        dynamic = renderer(cam, scene.dynamic_at(frame['time']), PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
        static, empty = observation_masks(rgba, max(2, 1600//100))
        masks = {'dynamic': dynamic_mask(rgba), 'static': static}
        row = {'frame': int(Path(frame['file_path']).stem), 'PSNR': float(-10*torch.log10((pred-gt).square().mean().clamp_min(1e-12)))}
        for label, mask in masks.items():
            m = torch.tensor(mask, device='cuda')
            row[label+'_PSNR'] = float(-10*torch.log10((pred[:, m]-gt[:, m]).square().mean().clamp_min(1e-12)))
        row['dynamic_alpha'] = float(dynamic['alpha'].squeeze()[torch.tensor(masks['dynamic'], device='cuda')].mean())
        row['dynamic_leakage_on_static'] = float(dynamic['alpha'].squeeze()[torch.tensor(static, device='cuda')].mean())
        row['dynamic_alpha_on_empty'] = float(dynamic['alpha'].squeeze()[torch.tensor(empty, device='cuda')].mean())
        rows.append(row)
    return {'count': len(rows), 'frames': rows,
        'metrics': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'frame'},
        'actual_original_GT_not_neutralized_target': True, 'test_RGB_read': False}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--enable-background-residual', action='store_true')
    p.add_argument('--full', action='store_true'); p.add_argument('--tag', default='screen'); args = p.parse_args()
    if not args.enable_background_residual:
        raise SystemExit('Background residual evaluation is opt-in')
    plan = verify(); torch.set_num_threads(4)
    renderer, scene, _, parent, _ = load_parent()
    paths = {'S3U_parent': PARENT}; states = {'S3U_parent': parent}
    for arm in plan['arms']:
        path = WORK/'formal'/arm/'checkpoint.pt'
        assert sha(path) == read(path.with_name('checkpoint.json'))['sha256']
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['iteration'] == plan['actual_updates'] and state['protocol_sha256'] == sha(EXP/'protocol.json')
        paths[arm] = path; states[arm] = state
    out = OUT/args.tag; assert not (out/'receipt.json').exists(); out.mkdir(parents=True, exist_ok=True)
    results = {arm: validation(renderer, scene, state) for arm, state in states.items()}
    write(EXP/f'{args.tag}_validation.json', results)
    meta = read(DATA/'transforms_train.json'); frames = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    times = plan['full_acceptance_times'] if args.full else plan['screen_times']
    summaries = {}; cameras = {}; html = []
    for cam_id in plan['fixed_cameras']:
        cam = camera(frames[cam_id], meta, 1600); cameras[str(cam_id)] = fingerprint(cam)
        all_rows = {arm: [] for arm in states}; panels = []
        for n in times:
            values = {}; alphas = []
            for arm, state in states.items():
                scene.load_state_dict(state['model']); t = (n-1)/199
                package = scene.render(renderer, cam, t)
                dynamic = renderer(cam, scene.dynamic_at(t), PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
                alpha = dynamic['alpha'].squeeze().cpu().numpy().clip(0, 1)
                values[arm] = (pixels(package['render']), pixels(dynamic['render']), alpha); alphas.append(alpha)
                proxy = cv2.resize(alpha, (400, 225), interpolation=cv2.INTER_AREA)
                all_rows[arm].append({'frame': n, 'shape_proxies': shape_proxies(proxy)})
            union = np.maximum.reduce(alphas) > .05; ys, xs = np.nonzero(union)
            box = (max(0, int(xs.min())-30), max(0, int(ys.min())-30), min(1600, int(xs.max())+31), min(900, int(ys.max())+31)) if len(xs) else (0, 0, 1600, 900)
            crops = []
            if not args.full or n in plan['screen_times']:
                for arm, (rgb, dyn_rgb, alpha) in values.items():
                    destination = out/arm/f'fixed{cam_id}'; destination.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(destination/f't{n:04d}_native.png')
                    Image.fromarray(dyn_rgb).save(destination/f't{n:04d}_dynamic.png')
                    Image.fromarray((alpha*255).round().astype(np.uint8)).save(destination/f't{n:04d}_alpha.png')
                    a, b, c, d = box; crops.append(tile(rgb[b:d, a:c], f'{arm}; c{cam_id}; t{n}'))
                height = max(len(v) for v in crops); row = np.zeros((height, 480*len(crops), 3), dtype=np.uint8)
                for i, value in enumerate(crops):
                    row[:len(value), i*480:(i+1)*480] = value
                panels.append(row)
            assert fingerprint(cam) == cameras[str(cam_id)]
        canvas = Image.new('RGB', (480*len(states), sum(len(v) for v in panels)), '#171a22'); y = 0
        for row in panels:
            canvas.paste(Image.fromarray(row), (0, y)); y += len(row)
        sheet = out/f'fixed{cam_id}_comparison.png'; canvas.save(sheet)
        html.append(f'<h2>Fixed camera{cam_id}</h2><img src="{sheet.name}">')
        summaries[str(cam_id)] = {arm: summarize(rows) for arm, rows in all_rows.items()}
        write(out/f'fixed{cam_id}_metrics.json', all_rows)
        print('BACKGROUND_RESIDUAL_VIEW_EVALUATED', cam_id, len(times), flush=True)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Background residual comparison</title><style>body{background:#171a22;color:#eee;font:16px system-ui}img{max-width:100%}</style><h1>Actual native RGB; shared crop per time across models</h1><p>Training reference intervention only. No unseen-view GT; images require actual review.</p>'+'\n'.join(html), encoding='utf-8')
    write(out/'receipt.json', {'models': {arm: {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'updates': states[arm]['iteration']} for arm, path in paths.items()},
        'width': 1600, 'fixed_camera_fingerprints': cameras, 'times': times, 'validation': results,
        'shape_summary': summaries, 'test_RGB_read': False, 'no_postprocessing_repair': True,
        'source_freeze_sha256': sha(EXP/'source_freeze.json'), 'acceptance': 'pending_actual_image_review'})
    write(EXP/f'{args.tag}_evaluation.json', read(out/'receipt.json')); verify()


if __name__ == '__main__':
    main()
