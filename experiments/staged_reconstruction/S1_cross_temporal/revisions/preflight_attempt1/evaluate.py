"""Native matched-camera comparison; inferred masks are never geometry GT."""
import hashlib
import subprocess
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..static_gaussian import StaticGaussian
from ..train_static import setup, camera, render_static
from ..automatic_static.discover import rgb
from .prepare import REPORT, RUNS, OUT, verify, region_confidence
from .core import summary


def pixels(value):
    return (value.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype('uint8')


def camera_hash(cam):
    value = cam.world_view_transform.detach().cpu().numpy().tobytes()+cam.full_proj_transform.detach().cpu().numpy().tobytes()
    return hashlib.sha256(value+str((cam.FoVx, cam.FoVy, cam.image_width, cam.image_height)).encode()).hexdigest()


def metric(package, target, mask):
    mask = torch.from_numpy(np.asarray(Image.fromarray(mask).resize((1600, 900), Image.Resampling.NEAREST))).cuda().bool()
    if not mask.any(): return None
    prediction = package['render'].clamp(0, 1); alpha = package['alpha'].squeeze()
    return {'pixels': int(mask.sum()), 'PSNR': float(-10*torch.log10((prediction[:, mask]-target[:, mask]).square().mean().clamp_min(1e-12))),
            'alpha_mean': float(alpha[mask].mean()), 'alpha_below_05': float((alpha[mask] < .5).float().mean())}


@torch.no_grad()
def main():
    verify(); assert not (REPORT/'exports.json').exists(), 'Preserve previous evaluation'
    protocol = read(REPORT/'protocol.json'); discovery = read(REPORT/'discovery.json')
    renderer = setup(); torch.set_num_threads(4); cv2.setNumThreads(2)
    models, checkpoints, fingerprints, states = {}, {}, {}, {}
    for arm in ['S1_A', 'fixed', 'progressive']:
        source = ROOT/'runs/pidg_staged_reconstruction/S1_automatic' if arm == 'S1_A' else RUNS
        checkpoint = source/'formal/checkpoint.pt' if arm == 'S1_A' else source/'formal'/arm/'checkpoint.pt'
        if arm != 'S1_A':
            audit = read(REPORT/f'formal_{arm}_audit.json')
            assert audit['passed'] and audit['iteration'] == 12000 and sha(checkpoint) == audit['checkpoint_sha256']
        model = StaticGaussian(source/'initialization.npz')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False); model.load_state_dict(state['model'])
        models[arm] = model; checkpoints[arm] = checkpoint; fingerprints[arm] = sha(checkpoint); states[arm] = state
    # Same common initial bank and camera sampling across the new arms.
    starts = [torch.load(RUNS/'formal'/arm/'initial.pt', map_location='cpu', weights_only=False) for arm in ['fixed', 'progressive']]
    assert all(torch.equal(starts[0]['model'][n], starts[1]['model'][n]) for n in starts[0]['model'])
    sequences = [[read_row['frame_index'] for line in (RUNS/'formal'/arm/'sampling.jsonl').read_text().splitlines()
                  for read_row in [__import__('json').loads(line)]] for arm in ['fixed', 'progressive']]
    assert sequences[0] == sequences[1] and len(sequences[0]) == 12000
    old_masks = np.load(ROOT/'runs/pidg_staged_reconstruction/S1_automatic/inferred_masks.npz')
    geometry = np.load(RUNS/'geometric_confidence.npz')
    rows, panels, exported = [], [], []
    selected = [2, 27, 43, 86, 166, 188, 200]; camera_fingerprints = {}
    for split in ['train', 'val']:
        meta = read(DATA/'transforms_train.json' if split == 'train' else ROOT/'data/derived/kinematic_resplit/transforms_val.json')
        for frame in meta['frames']:
            number = int(Path(frame['file_path']).stem); name = str(number)
            gt = torch.from_numpy(rgb(frame, 1600)).permute(2, 0, 1).cuda(); cam = camera(frame, meta, 1600)
            fingerprint = camera_hash(cam)
            if split == 'train': masks = {'common_S1_A_train_proxy': old_masks[name], 'common_candidate_train_proxy': geometry[name] >= .5}
            else:
                _, inferred = region_confidence(frame, rgb(frame, 400), discovery['anchors'], discovery['incompatible'], meta, protocol)
                masks = {'val_inferred_geometric_region_not_GT': inferred >= .5}
            row = {'split': split, 'frame': number, 'models': {}}
            rendered = {}
            for arm, model in models.items():
                p = render_static(renderer, cam, model); rendered[arm] = pixels(p['render'])
                assert camera_hash(cam) == fingerprint
                row['models'][arm] = {label: metric(p, gt, mask) for label, mask in masks.items()}
                if split == 'train' and number in selected:
                    path = OUT/f'{arm}_t{number:04d}_native.png'; Image.fromarray(rendered[arm]).save(path); exported.append(path)
            rows.append(row)
            if split == 'train' and number in selected:
                camera_fingerprints[name] = fingerprint
                panel = Image.new('RGB', (1600, 250), '#161922')
                values = [('Observed RGB (includes motion)', pixels(gt)), ('S1-A old12k', rendered['S1_A']),
                          ('Common init fixed confidence12k', rendered['fixed']), ('Progressive confidence12k', rendered['progressive'])]
                for i, (label, value) in enumerate(values):
                    panel.paste(Image.fromarray(value).resize((400, 225), Image.Resampling.LANCZOS), (i*400, 25))
                    ImageDraw.Draw(panel).text((i*400+6, 5), f't{number} {label}', fill='white')
                panels.append(np.asarray(panel))
    path = OUT/'native_comparison.png'; Image.fromarray(np.concatenate(panels)).save(path); exported.append(path)
    # Confidence overlays are training diagnostics; they do not certify staticity.
    frames = read(DATA/'transforms_train.json')['frames']; overlays = []
    for number in selected:
        frame = next(f for f in frames if int(Path(f['file_path']).stem) == number)
        panel = Image.new('RGB', (1200, 250), '#161922')
        for i, (arm, c) in enumerate([('Initial / fixed', states['fixed']['confidence'][str(number)]),
            ('Progressive final', states['progressive']['confidence'][str(number)]), ('Candidate upper bound', geometry[str(number)])]):
            im = (rgb(frame, 400)*255).round().astype('uint8'); positive = c >= .5
            im[positive] = .35*im[positive]+.65*np.array([0, 255, 100])
            panel.paste(Image.fromarray(im), (i*400, 25)); ImageDraw.Draw(panel).text((i*400+6, 5), f't{number} {arm}', fill='white')
        overlays.append(np.asarray(panel))
    Image.fromarray(np.concatenate(overlays)).save(OUT/'confidence_final.png'); exported.append(OUT/'confidence_final.png')
    from scene_PIDG.gaussian_model import GaussianModel
    plys = {}
    for arm in ['fixed', 'progressive']:
        native = GaussianModel(3); native.active_sh_degree = 3
        for name, parameter in models[arm].named_parameters(): setattr(native, name, parameter)
        native.dynamic_mask = torch.zeros(len(models[arm].get_xyz), dtype=torch.bool, device='cuda')
        path = OUT/f'{arm}_12000.ply'; native.save_ply(str(path)); plys[arm] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}
    all_meta = read(ROOT/'data/PIDG/rubber_duck/transforms_all.json')
    all_frames = sorted(all_meta['frames'], key=lambda f: int(Path(f['file_path'].replace('\\', '/')).stem))
    video = OUT/'original_trajectory_comparison.mp4'
    assert not video.exists()
    with (OUT/'original_trajectory_comparison.encode.log').open('wb') as log:
        process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-n', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '800x500', '-r', '20', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-profile:v', 'main',
            '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        for frame in all_frames:
            number = int(Path(frame['file_path'].replace('\\', '/')).stem)
            cam = camera(frame, all_meta, 1600); fingerprint = camera_hash(cam)
            panel = Image.new('RGB', (800, 500), '#161922')
            images = [('Observed RGB (includes motion)', (rgb(frame, 400)*255).round().astype('uint8'))]
            images += [(arm+' static12k', pixels(render_static(renderer, cam, model)['render'])) for arm, model in models.items()]
            assert camera_hash(cam) == fingerprint
            for i, (label, value) in enumerate(images):
                x, y = (i % 2)*400, (i//2)*250
                panel.paste(Image.fromarray(value).resize((400, 225), Image.Resampling.LANCZOS), (x, y+25))
                ImageDraw.Draw(panel).text((x+6, y+5), f't{number} {label}', fill='white')
            process.stdin.write(np.asarray(panel).tobytes())
        process.stdin.close(); assert process.wait(timeout=60) == 0
    cap = cv2.VideoCapture(str(video)); count = 0
    while cap.read()[0]: count += 1
    cap.release(); assert count == 200
    aggregate = {}
    for label in ['common_S1_A_train_proxy', 'common_candidate_train_proxy', 'val_inferred_geometric_region_not_GT']:
        aggregate[label] = {}
        for arm in models:
            values = [r['models'][arm][label] for r in rows if label in r['models'][arm] and r['models'][arm][label] is not None]
            aggregate[label][arm] = {'nonempty_frames': len(values), 'pixels_total': sum(v['pixels'] for v in values),
                **{k: float(np.mean([v[k] for v in values])) if values else None for k in ['PSNR', 'alpha_mean', 'alpha_below_05']}}
    assert all(sha(checkpoints[a]) == fingerprints[a] for a in models); verify()
    result = {'stage': 'S1-CT', 'updates_per_arm': 12000, 'static_points_new_arms': len(models['fixed'].get_xyz),
        'matched_initial_model_and_camera_sequence': True, 'checkpoint_sha256': fingerprints, 'metrics': aggregate, 'frames': rows,
        'confidence_summary': {a: summary(states[a]['confidence']) for a in ['fixed', 'progressive']},
        'confidence_history': states['progressive']['confidence_history'], 'fixed_camera_fingerprints': camera_fingerprints,
        'images': {p.relative_to(ROOT).as_posix(): sha(p) for p in exported}, 'PLY': plys,
        'video': {'path': video.relative_to(ROOT).as_posix(), 'sha256': sha(video), 'decoded_frames': count,
                  'codec': 'H264 Main / yuv420p / faststart', 'layout': 'Observed RGB / S1-A / fixed / progressive'},
        'independent_geometry_or_static_labels_GT': False, 'validation_pixels_are_inferred_not_GT': True,
        'full_static_reconstruction_claimed': False, 'static_fixed_camera_has_no_time_variation_by_construction': True,
        'manual_visual_review_is_separate': True}
    write(REPORT/'exports.json', result)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>S1-CT comparison</title>'
        '<style>body{background:#161922;color:white;font:17px system-ui;max-width:1600px;margin:30px auto}img,video{max-width:100%}a{color:#9cf}</style>'
        '<h1>Cross-temporal confidence: matched12k static 3DGS arms</h1>'
        '<p>Geometry-conditioned confidence, no known static color. Partial reconstruction; inferred regions are not geometry GT.</p>'
        '<p><a href="progressive_12000.ply">Progressive3DGS PLY</a> | <a href="fixed_12000.ply">Fixed control PLY</a></p>'
        '<h2>Observed RGB / S1-A / fixed / progressive, same cameras</h2><img src="native_comparison.png">'
        '<h2>Confidence: initial / final / geometric upper bound</h2><img src="confidence_final.png">'
        '<h2>Original camera trajectory,200 frames</h2><video controls preload="metadata" poster="progressive_t0086_native.png">'
        '<source src="original_trajectory_comparison.mp4" type="video/mp4"></video>'
        '<p>Fixed-camera static fields do not change with time; native86/188 images are provided above.</p>', encoding='utf8')
    print('CROSS_TEMPORAL_EXPORTED', aggregate, flush=True)


if __name__ == '__main__': main()
