"""Native cross-view screening with dynamic-only diagnostics, no test selection."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ..prepare_data import ROOT, DATA, REPORT, read, write, sha
from ..train_static import camera, PIPE
from ..stage2.train import image
from ..union.core import dynamic_mask
from variants.view_completeness.metrics import shape_proxies, summarize
from .prepare import EXP, WORK, PARENT, PARENT_SHA, load_parent

OUT = ROOT/'output/staged_reconstruction/motion_integrity'


def pixels(tensor):
    return (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)


def fingerprint(cam):
    return hashlib.sha256(cam.world_view_transform.cpu().numpy().tobytes()+cam.full_proj_transform.cpu().numpy().tobytes()).hexdigest()


def load_states():
    states = {'S3U_parent': torch.load(PARENT, map_location='cpu', weights_only=False)}
    paths = {'S3U_parent': PARENT}
    for arm in read(EXP/'protocol.json')['arms']:
        path = WORK/'formal'/arm/'checkpoint.pt'
        if not path.exists():
            continue
        assert sha(path) == read(path.with_name('checkpoint.json'))['sha256']
        states[arm] = torch.load(path, map_location='cpu', weights_only=False)
        paths[arm] = path
    return states, paths


@torch.no_grad()
def validation(renderer, scene, state):
    scene.load_state_dict(state['model'])
    meta = read(ROOT/'data/derived/kinematic_resplit/transforms_val.json')
    records = []
    for frame in meta['frames']:
        rgba = image(frame, 1600)
        gt = torch.tensor(rgba[..., :3]*rgba[..., 3:4], device='cuda').permute(2, 0, 1)
        cam = camera(frame, meta, 1600)
        pred = scene.render(renderer, cam, frame['time'])['render'].clamp(0, 1)
        dyn = renderer(cam, scene.dynamic_at(frame['time']), PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
        mask = torch.tensor(dynamic_mask(rgba), device='cuda')
        records.append({'frame': int(Path(frame['file_path']).stem),
                        'PSNR': float(-10*torch.log10((pred-gt).square().mean().clamp_min(1e-12))),
                        'dynamic_PSNR': float(-10*torch.log10((pred[:, mask]-gt[:, mask]).square().mean().clamp_min(1e-12))),
                        'dynamic_alpha': float(dyn['alpha'].squeeze()[mask].mean())})
    return {'count': len(records), 'frames': records,
            'metrics': {k: float(np.mean([r[k] for r in records])) for k in records[0] if k != 'frame'},
            'test_RGB_read': False}


def tile(value, label, width=480):
    im = Image.fromarray(value)
    height = round(im.height*width/im.width)
    canvas = Image.new('RGB', (width, height+28), '#171a22')
    canvas.paste(im.resize((width, height), Image.Resampling.LANCZOS), (0, 28))
    ImageDraw.Draw(canvas).text((6, 5), label, fill='white', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15))
    return np.asarray(canvas)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--enable-motion-integrity', action='store_true')
    p.add_argument('--full', action='store_true')
    p.add_argument('--tag', default='screen')
    args = p.parse_args()
    if not args.enable_motion_integrity:
        raise SystemExit('Motion integrity evaluation is opt-in')
    torch.set_num_threads(4)
    renderer, scene, _, _, _ = load_parent()
    states, paths = load_states()
    plan = read(EXP/'protocol.json')
    out = OUT/args.tag
    assert not (out/'receipt.json').exists(), 'Keep evaluation versions'
    out.mkdir(parents=True, exist_ok=True)
    meta = read(DATA/'transforms_train.json')
    by_id = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    times = plan['full_acceptance_times'] if args.full else plan['screen_times']
    summaries, fingerprints, html = {}, {}, []
    results = {arm: validation(renderer, scene, state) for arm, state in states.items()}
    for cam_id in plan['fixed_cameras']:
        cam = camera(by_id[cam_id], meta, 1600)
        fingerprints[str(cam_id)] = fingerprint(cam)
        all_rows = {arm: [] for arm in states}
        panels = []
        for number in times:
            crops, alphas, native = {}, [], {}
            for arm, state in states.items():
                scene.load_state_dict(state['model'])
                t = (number-1)/199
                dynamic = scene.dynamic_at(t)
                composite = scene.render(renderer, cam, t)
                dyn = renderer(cam, dynamic, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
                rgb, dyn_rgb = pixels(composite['render']), pixels(dyn['render'])
                alpha = dyn['alpha'].squeeze().cpu().numpy().clip(0, 1)
                small = cv2.resize(alpha, (400, 225), interpolation=cv2.INTER_AREA)
                all_rows[arm].append({'frame': number, 'shape_proxies': shape_proxies(small)})
                alphas.append(alpha)
                native[arm] = (rgb, dyn_rgb, alpha)
            union = np.maximum.reduce(alphas) > .05
            ys, xs = np.nonzero(union)
            box = (max(0, int(xs.min())-30), max(0, int(ys.min())-30), min(1600, int(xs.max())+31), min(900, int(ys.max())+31)) if len(xs) else (0, 0, 1600, 900)
            for arm, (rgb, dyn_rgb, alpha) in native.items():
                destination = out/arm/f'fixed{cam_id}'
                destination.mkdir(parents=True, exist_ok=True)
                if not args.full or number in plan['screen_times']:
                    Image.fromarray(rgb).save(destination/f't{number:04d}_native.png')
                    Image.fromarray(dyn_rgb).save(destination/f't{number:04d}_dynamic.png')
                    Image.fromarray((alpha*255).round().astype(np.uint8)).save(destination/f't{number:04d}_alpha.png')
                    a, b, c, d = box
                    crops[arm] = tile(rgb[b:d, a:c], f'{arm}; c{cam_id}; t{number}')
            if crops:
                # A common crop preserves the same scale across arms; padding is only for layout.
                height = max(v.shape[0] for v in crops.values())
                parts = []
                for value in crops.values():
                    padded = np.zeros((height, value.shape[1], 3), dtype=np.uint8)
                    padded[:len(value)] = value
                    parts.append(padded)
                panels.append(np.concatenate(parts, axis=1))
            assert fingerprint(cam) == fingerprints[str(cam_id)]
        # Each row may have a different common crop aspect ratio.
        canvas = Image.new('RGB', (480*len(states), sum(len(row) for row in panels)), '#171a22')
        y = 0
        for row in panels:
            canvas.paste(Image.fromarray(row), (0, y)); y += len(row)
        sheet = out/f'fixed{cam_id}_comparison.png'
        canvas.save(sheet)
        html.append(f'<h2>Fixed camera {cam_id}</h2><img src="{sheet.name}">')
        summaries[str(cam_id)] = {arm: summarize(rows) for arm, rows in all_rows.items()}
        write(out/f'fixed{cam_id}_metrics.json', all_rows)
        print('MOTION_VIEW_EVALUATED', cam_id, len(times), flush=True)
    html.insert(0, '<!doctype html><meta charset="utf-8"><title>Motion integrity</title><style>body{background:#151821;color:#eee;font:16px system-ui}img{max-width:100%}</style><h1>Literature-guided motion integrity</h1><p>Native1600 renders, common dynamic crop across arms. No unseen-view GT. Rendered evidence requires visual review; metrics alone do not establish completeness.</p>')
    (out/'index.html').write_text('\n'.join(html), encoding='utf-8')
    write(out/'receipt.json', {'models': {arm: {'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'updates': states[arm]['iteration']} for arm, p in paths.items()},
          'fixed_camera_fingerprints': fingerprints, 'times': times, 'width': 1600, 'proxy_width': 400,
          'validation': results, 'shape_summary': summaries, 'test_RGB_read': False,
          'no_postprocessing_repair': True, 'independent_new_view_GT': False, 'acceptance': 'pending_visual_review'})
    write(EXP/f'{args.tag}_evaluation.json', read(out/'receipt.json'))
    assert sha(PARENT) == PARENT_SHA


if __name__ == '__main__':
    main()
