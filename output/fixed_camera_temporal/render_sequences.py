"""Read-only native render of frozen camera / varying time, with moving control."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'variants/association_training/code'))
os.environ['PIDG_HASH_SOURCE_DIR'] = str(ROOT / 'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR'] = str(ROOT / 'third_party/PIDG/tmp_build')
from variants.association_training.evaluate import load, render_call
from variants.shape_coherence.observations import camera
from variants.flow_comparison.temporal import warp_at
from variants.metric_tuning.runtime import sha256, read_json, write_json

OUT = ROOT / 'output/fixed_camera_temporal'
REPORT = ROOT / 'experiments/fixed_camera_temporal'
MODELS = {
    'C1_original_50k': ROOT / 'runs/pidg_color_repair/C1_repaired/rubber_duck',
    'T1_association_52k': ROOT / 'runs/pidg_association_training/pilot/T1_association/rubber_duck',
    'O1_flow_on_50k': ROOT / 'runs/pidg_c1_flow_ablation/O1_C1_flow_on/rubber_duck'}
POSES = [2, 86, 188]
SELECTED = [2, 27, 43, 74, 86, 108, 172, 200]
VIEWS = [f'fixed_{p:04d}' for p in POSES] + ['moving_control']


def fingerprint(cam):
    result = hashlib.sha256()
    for name in ['world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center']:
        result.update(getattr(cam, name).detach().cpu().numpy().tobytes())
    result.update(json.dumps([cam.FoVx, cam.FoVy, cam.image_width, cam.image_height]).encode())
    return result.hexdigest()


class Video:
    def __init__(self, path, width, height):
        self.log = path.with_suffix('.encode.log').open('wb')
        self.proc = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-v', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '25', '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-threads', '2',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def add(self, array):
        self.proc.stdin.write(np.ascontiguousarray(array).tobytes())

    def close(self):
        self.proc.stdin.close()
        code = self.proc.wait(timeout=60)
        self.log.close()
        assert code == 0, code


def shape_stats(mask):
    labels, count = ndimage.label(mask, np.ones((3, 3)))
    areas = np.bincount(labels.ravel())[1:]
    large = np.sort(areas[areas >= 20])[::-1]
    # Hole counts are computed on the original binary mask, never repaired RGB.
    holes = ndimage.binary_fill_holes(mask) & ~mask
    hl, _ = ndimage.label(holes, np.ones((3, 3)))
    ha = np.bincount(hl.ravel())[1:]
    return {'area': int(mask.sum()), 'components_ge20px': len(large),
            'component_areas_ge20px': large.astype(int).tolist(),
            'small_component_area': int(areas[areas < 20].sum()),
            'holes_ge4px': int((ha >= 4).sum()), 'hole_area_ge4px': int(ha[ha >= 4].sum()),
            'border_pixels': int(mask[0].sum()+mask[-1].sum()+mask[1:-1, 0].sum()+mask[1:-1, -1].sum())}


def descriptive(rgb, alpha):
    value = rgb.astype(np.float32)/255
    yellow = ((value[..., 0] > .15) & (value[..., 1] > .13)
              & (np.minimum(value[..., 0], value[..., 1]) > value[..., 2]*1.35 + .03)
              & (alpha > .5))
    return {'alpha_05': shape_stats(alpha > .5), 'alpha_09': shape_stats(alpha > .9),
            'yellow_appearance': shape_stats(yellow)}


def metadata_all():
    parts = [read_json(ROOT / f'data/derived/kinematic_resplit/transforms_{s}.json')
             for s in ['train', 'val', 'test']]
    frames = {int(Path(f['file_path']).stem): f for p in parts for f in p['frames']}
    assert len(frames) == 200 and sorted(frames) == list(range(1, 201))
    assert all(abs(f['time']-(n-1)/199) < 1e-9 for n, f in frames.items())
    train = {int(Path(f['file_path']).stem) for f in parts[0]['frames']}
    assert set(POSES) <= train
    return parts[0], frames


def preflight(name, g, d, meta, frames):
    cam = camera(frames[86], meta, 1600)
    before = fingerprint(cam)
    w = warp_at(d, g.get_xyz.detach(), 85/199)
    pkg = render_call(cam, g, w)
    rgb = pkg['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    quant = (rgb*255).round().astype('uint8')
    assert np.isfinite(rgb).all() and pkg['alpha'].isfinite().all()
    old = {'C1_original_50k': 'C1_original', 'T1_association_52k': 'T1_association'}.get(name)
    check = {'camera_fingerprint': before, 'finite': True}
    if old:
        path = ROOT / f'output/association_training/{old}/native_GT_Render0086.png'
        ref = np.array(Image.open(path).convert('RGB'))[:, 1600:]
        error = np.abs(quant.astype(int)-ref.astype(int))
        check.update(reference=str(path.relative_to(ROOT)), reference_sha256=sha256(path),
                     native_diagonal_max_LSB_error=int(error.max()), mean_LSB_error=float(error.mean()))
        assert error.max() <= 1, check
    assert fingerprint(cam) == before
    Image.fromarray(quant).save(OUT / f'preflight_{name}.png')
    return check


def contacts(name):
    for view in VIEWS:
        canvas = Image.new('RGB', (1600, 960), '#171a21')
        draw = ImageDraw.Draw(canvas)
        for k, n in enumerate(SELECTED):
            x, y = (k % 4)*400, (k // 4)*480
            draw.text((x+8, y+5), f'{name} | {view} | time frame {n:03d}', fill='white')
            base = OUT/name/view
            canvas.paste(Image.open(base/'small'/f'rgb{n:04d}.png'), (x, y+26))
            canvas.paste(Image.open(base/'small'/f'alpha{n:04d}.png').convert('RGB'), (x, y+251))
        canvas.save(OUT/name/f'{view}_RGB_alpha_contact.png')


@torch.no_grad()
def run_model(name, directory, meta, frames, preflight_only=False):
    os.environ['PIDG_METRIC_CONFIG'] = str(directory/'metric_config.json')
    from variants.color_repair.color import _read
    _read.cache_clear()
    g, d, checkpoint = load(directory)
    result = {'checkpoint': checkpoint, 'optimizer_updates': 0,
              'preflight': preflight(name, g, d, meta, frames)}
    write_json(REPORT/f'{name}_preflight.json', result)
    print(name, 'native preflight passed', result['preflight'], flush=True)
    if preflight_only:
        return result
    fixed = {f'fixed_{p:04d}': camera(frames[p], meta, 1600) for p in POSES}
    hashes = {view: fingerprint(cam) for view, cam in fixed.items()}
    videos = {}; rows = {view: [] for view in VIEWS}
    for view in VIEWS:
        base = OUT/name/view
        (base/'small').mkdir(parents=True, exist_ok=True)
        (base/'native').mkdir(exist_ok=True)
        videos[view] = Video(OUT/name/(view+'_RGB.mp4'), 800, 480)
    start = time.time()
    try:
        for n in range(1, 201):
            w = warp_at(d, g.get_xyz.detach(), (n-1)/199)
            for view in VIEWS:
                cam = camera(frames[n], meta, 1600) if view == 'moving_control' else fixed[view]
                pkg = render_call(cam, g, w)
                rgb = pkg['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                alpha = pkg['alpha'].squeeze().clamp(0, 1).cpu().numpy()
                assert rgb.shape == (900, 1600, 3) and alpha.shape == (900, 1600)
                assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
                native = Image.fromarray((rgb*255).round().astype('uint8'))
                alpha_im = Image.fromarray((alpha*255).round().astype('uint8'))
                small = native.resize((400, 225), Image.Resampling.LANCZOS)
                # Compute stats on area-averaged native alpha, not a new low-resolution splat.
                alpha_small = np.asarray(Image.fromarray(alpha).resize((400, 225), Image.Resampling.BOX))
                base = OUT/name/view
                small.save(base/'small'/f'rgb{n:04d}.png')
                alpha_im.resize((400, 225), Image.Resampling.BOX).save(base/'small'/f'alpha{n:04d}.png')
                if n in SELECTED:
                    native.save(base/'native'/f'rgb{n:04d}.png')
                    alpha_im.save(base/'native'/f'alpha{n:04d}.png')
                row = {'frame': n, 'time': (n-1)/199, **descriptive(np.asarray(small), alpha_small)}
                rows[view].append(row)
                panel = Image.new('RGB', (800, 480), '#171a21')
                panel.paste(native.resize((800, 450), Image.Resampling.LANCZOS), (0, 30))
                ImageDraw.Draw(panel).text((10, 8), f'{name} | {view} | time {n:03d}/200 ({(n-1)/199:.4f})', fill='white')
                videos[view].add(np.asarray(panel))
            if n % 10 == 0:
                assert all(fingerprint(cam) == hashes[v] for v, cam in fixed.items())
                print(name, n, '/ 200 times x 4 cameras;', round(time.time()-start, 1), 's', flush=True)
    finally:
        for video in videos.values():
            video.close()
    result.update(fixed_camera_fingerprints_before=hashes,
                  fixed_camera_fingerprints_after={v: fingerprint(c) for v, c in fixed.items()},
                  elapsed_seconds=time.time()-start, frames=rows)
    assert checkpoint['sha256'] == sha256(directory/'checkpoints/resume.pt')
    result['checkpoint_unchanged'] = True
    write_json(REPORT/f'{name}_results.json', result)
    contacts(name)
    del g, d, w, pkg
    torch.cuda.empty_cache()
    return result


def comparisons():
    for view in VIEWS:
        video = Video(OUT/f'{view}_comparison_RGB_alpha.mp4', 1200, 510)
        try:
            for n in range(1, 201):
                panel = Image.new('RGB', (1200, 510), '#171a21')
                draw = ImageDraw.Draw(panel)
                for k, name in enumerate(MODELS):
                    x = 400*k
                    draw.text((x+8, 7), f'{name} | {view} | frame {n:03d}', fill='white')
                    base = OUT/name/view/'small'
                    panel.paste(Image.open(base/f'rgb{n:04d}.png'), (x, 30))
                    draw.text((x+8, 262), 'Opacity alpha (white = opaque)', fill='white')
                    panel.paste(Image.open(base/f'alpha{n:04d}.png').convert('RGB'), (x, 285))
                video.add(np.asarray(panel))
                if n in SELECTED:
                    panel.save(OUT/f'{view}_comparison_t{n:04d}.png')
        finally:
            video.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    meta, frames = metadata_all()
    protected = {str(d/'checkpoints/resume.pt'): sha256(d/'checkpoints/resume.pt') for d in MODELS.values()}
    inputs = [Path(__file__), REPORT/'plan.json', ROOT/'variants/association_training/evaluate.py',
              ROOT/'variants/shape_coherence/observations.py', ROOT/'variants/unified_flow/camera_contract.py',
              ROOT/'variants/flow_comparison/temporal.py']
    inputs += [ROOT/f'data/derived/kinematic_resplit/transforms_{s}.json' for s in ['train', 'val', 'test']]
    inputs += [d/f for d in MODELS.values() for f in ['model_config.py', 'metric_config.json']]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in inputs}
    for name, directory in MODELS.items():
        run_model(name, directory, meta, frames, args.preflight_only)
    assert all(sha256(Path(p)) == h for p, h in protected.items())
    assert all(sha256(ROOT/p) == h for p, h in source_hashes.items())
    if not args.preflight_only:
        comparisons()
    write_json(REPORT/('preflight_verification.json' if args.preflight_only else 'render_verification.json'),
               {'checkpoint_sha256_unchanged': protected, 'source_sha256_unchanged': source_hashes,
                'optimizer_updates': 0, 'fixed_pose_frames': POSES, 'time_count': 200,
                'native_resolution': [1600, 900], 'model_count': 3, 'new_view_ground_truth_available': False})
    print('COMPLETED', 'preflight' if args.preflight_only else 'all renders and videos', flush=True)


if __name__ == '__main__':
    main()
