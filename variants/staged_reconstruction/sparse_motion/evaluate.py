"""Native screen of the sparse representation against the same S3-U/I0/I1."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from ..prepare_data import ROOT, RUNS, DATA, read, write, sha
from ..train_static import camera, PIPE
from ..motion_integrity.prepare import load_parent, PARENT, PARENT_SHA
from ..motion_integrity.evaluate import pixels, tile, fingerprint, validation
from variants.view_completeness.metrics import shape_proxies, summarize
from .prepare import EXP, WORK
from .train import install

OUT = ROOT/'output/staged_reconstruction/sparse_motion'


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--enable-sparse-motion', action='store_true')
    p.add_argument('--full', action='store_true')
    p.add_argument('--tag', default='screen')
    args = p.parse_args()
    if not args.enable_sparse_motion:
        raise SystemExit('Sparse evaluation is opt-in')
    torch.set_num_threads(4)
    renderer, scene, _, _, _ = load_parent()
    plan = read(EXP/'protocol.json')
    original = scene.deforms[0]
    adapter = install(scene, plan)
    paths = {'S3U_parent': PARENT,
             'I0_control': RUNS/'motion_integrity/formal/I0_control/checkpoint.pt',
             'I1_curve_isometry': RUNS/'motion_integrity/formal/I1_curve_isometry/checkpoint.pt',
             'I2_sparse_motion': WORK/'formal/checkpoint.pt'}
    states = {}
    for arm, path in paths.items():
        assert sha(path) == read(path.with_name('checkpoint.json'))['sha256']
        states[arm] = torch.load(path, map_location='cpu', weights_only=False)
    assert states['I2_sparse_motion']['iteration'] == plan['actual_updates']
    assert states['I2_sparse_motion']['protocol_sha256'] == sha(EXP/'protocol.json')
    out = OUT/args.tag
    assert not (out/'receipt.json').exists(), 'Preserve evaluation versions'
    out.mkdir(parents=True, exist_ok=True)
    meta = read(DATA/'transforms_train.json')
    poses = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    times = plan['full_acceptance_times'] if args.full else plan['screen_times']
    summaries, val, hashes = {}, {}, {}
    for arm, state in states.items():
        scene.deforms[0] = adapter if arm == 'I2_sparse_motion' else original
        adapter.mix = min(1., state['iteration']/plan['motion_mix_ramp_updates'])
        scene.load_state_dict(state['model'])
        val[arm] = validation(renderer, scene, state)
        for cam_id in plan['fixed_cameras']:
            cam = camera(poses[cam_id], meta, 1600)
            assert hashes.setdefault(str(cam_id), fingerprint(cam)) == fingerprint(cam)
            folder = out/arm/f'fixed{cam_id}'
            folder.mkdir(parents=True, exist_ok=True)
            rows = []
            for number in times:
                t = (number-1)/199
                dynamic = scene.dynamic_at(t)
                rgb = pixels(scene.render(renderer, cam, t)['render'])
                dyn = renderer(cam, dynamic, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
                alpha = dyn['alpha'].squeeze().cpu().numpy().clip(0, 1)
                rows.append({'frame': number, 'shape_proxies': shape_proxies(cv2.resize(alpha, (400, 225), interpolation=cv2.INTER_AREA))})
                if not args.full or number in plan['screen_times']:
                    Image.fromarray(rgb).save(folder/f't{number:04d}_native.png')
                    Image.fromarray(pixels(dyn['render'])).save(folder/f't{number:04d}_dynamic.png')
                    Image.fromarray((alpha*255).round().astype(np.uint8)).save(folder/f't{number:04d}_alpha.png')
                assert fingerprint(cam) == hashes[str(cam_id)]
            summaries.setdefault(str(cam_id), {})[arm] = summarize(rows)
            write(folder/'metrics.json', rows)
            print('SPARSE_EVAL', arm, cam_id, len(times), flush=True)
    html = ['<!doctype html><meta charset="utf-8"><title>Sparse shared motion</title><style>body{font:16px system-ui;background:#151821;color:#eee}img{max-width:100%}</style><h1>S3-U | I0 | I1 | I2 sparse motion</h1><p>Native1600 renders, same crop across arms in each row. No image repair or new-view GT. Shape proxies cannot certify correctness.</p>']
    for cam_id in plan['fixed_cameras']:
        panels = []
        for number in times:
            if args.full and number not in plan['screen_times']:
                continue
            masks = [np.asarray(Image.open(out/arm/f'fixed{cam_id}/t{number:04d}_alpha.png')) for arm in paths]
            ys, xs = np.nonzero(np.maximum.reduce(masks) > 12)
            box = (max(0, int(xs.min())-30), max(0, int(ys.min())-30), min(1600, int(xs.max())+31), min(900, int(ys.max())+31)) if len(xs) else (0, 0, 1600, 900)
            parts = []
            for arm in paths:
                value = np.asarray(Image.open(out/arm/f'fixed{cam_id}/t{number:04d}_native.png').convert('RGB').crop(box))
                parts.append(tile(value, f'{arm}; c{cam_id}; t{number}'))
            panels.append(np.concatenate(parts, axis=1))
        sheet = Image.new('RGB', (480*len(paths), sum(len(p) for p in panels)), '#171a22')
        y = 0
        for panel in panels:
            sheet.paste(Image.fromarray(panel), (0, y)); y += len(panel)
        sheet.save(out/f'fixed{cam_id}_comparison.png')
        html.append(f'<h2>Fixed camera{cam_id}</h2><img src="fixed{cam_id}_comparison.png">')
    (out/'index.html').write_text('\n'.join(html), encoding='utf-8')
    receipt = {'models': {arm: {'checkpoint': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'actual_updates': states[arm]['iteration'],
                              'sparse_mix': 1. if arm == 'I2_sparse_motion' else 0.} for arm, path in paths.items()},
               'validation': val, 'shape_summary': summaries, 'camera_fingerprints': hashes, 'times': times,
               'native_width': 1600, 'proxy_width': 400, 'test_RGB_read': False,
               'new_view_GT': False, 'acceptance': 'pending_visual_review'}
    write(out/'receipt.json', receipt)
    write(EXP/f'{args.tag}_evaluation.json', receipt)
    assert sha(PARENT) == PARENT_SHA


if __name__ == '__main__':
    main()
