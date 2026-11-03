"""Read-only native evaluation. All region masks are conditional proxies, not GT."""
import json
import subprocess
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
import torch
from output.short_clip_diagnostic.camera_math import w2c, intrinsic, project
from ..prepare_data import ROOT, DATA, read, write, sha
from ..automatic_static.discover import rgb
from ..static_gaussian import StaticGaussian
from ..train_static import setup, camera, render_static
from ..cross_temporal_static.evaluate import pixels, camera_hash, metric
from ..cross_temporal_static.prepare import region_confidence
from ..cross_temporal_static.core import summary
from ..adaptive_static.model import AdaptiveGaussian
from ..adaptive_static.prepare import RUNS as PREVIOUS_RUNS
from .prepare import REPORT, RUNS, OUT, verify


def observed(frame, width):
    path = ROOT/'data/PIDG/rubber_duck'/(frame['file_path'].replace('\\', '/')+'.png')
    assert path.resolve().is_relative_to((ROOT/'data/PIDG/rubber_duck').resolve())
    with Image.open(path) as image:
        value = np.asarray(image.convert('RGBA').resize((width, width*9//16), Image.Resampling.BILINEAR), dtype=np.float32)/255
    return value[..., :3]*value[..., 3:4]


def new_inferred_mask(frame, image, meta, reservoir):
    """Z-buffered train-reservoir projections, 2px neighborhoods and RGB agreement.

    Validation RGB selects only the evaluation region. It never updates geometry,
    training confidence, topology or weights. This is not independent static GT.
    """
    uv, z = project(reservoir['xyz'], w2c(frame['transform_matrix'], 'metadata_cv'),
                    intrinsic(meta['camera_angle_x'], 'metadata', 400, 225))
    good = (z > .2)&np.isfinite(uv).all(1)&(uv[:, 0] >= 2)&(uv[:, 0] < 398)&(uv[:, 1] >= 2)&(uv[:, 1] < 223)
    indices = np.flatnonzero(good); xy = np.rint(uv[indices]).astype(int)
    depth = np.full((225, 400), np.inf)
    np.minimum.at(depth, (xy[:, 1], xy[:, 0]), z[indices])
    front = z[indices] <= depth[xy[:, 1], xy[:, 0]]+.025
    indices, xy = indices[front], xy[front]
    colors = np.zeros_like(image); support = np.zeros((225, 400), np.uint8)
    for index, (x, y) in zip(indices[np.argsort(-z[indices])], xy[np.argsort(-z[indices])]):
        cv2.circle(support, (int(x), int(y)), 2, 1, -1)
        cv2.circle(colors, (int(x), int(y)), 2, tuple(float(v) for v in reservoir['rgb'][index]), -1)
    return support.astype(bool)&(np.abs(colors-image).mean(2) <= .12)


def panel(images, number, width=400):
    result = Image.new('RGB', (width*2, (width*9//16+25)*2), '#161922')
    for i, (name, value) in enumerate(images):
        x, y = (i % 2)*width, (i//2)*(width*9//16+25)
        result.paste(Image.fromarray(value).resize((width, width*9//16), Image.Resampling.LANCZOS), (x, y+25))
        ImageDraw.Draw(result).text((x+5, y+5), f't{number} {name}', fill='white')
    return result


def encode(path, frames):
    assert not path.exists()
    with path.with_suffix('.encode.log').open('wb') as log:
        process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-n',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '800x500', '-r', '20', '-i', 'pipe:0', '-an',
            '-c:v', 'libx264', '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        count = 0
        for value in frames: process.stdin.write(np.asarray(value).tobytes()); count += 1
        process.stdin.close(); assert process.wait(timeout=60) == 0
    cap = cv2.VideoCapture(str(path)); decoded = 0
    while cap.read()[0]: decoded += 1
    cap.release(); assert decoded == count
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'decoded_frames': decoded,
            'codec': 'H264 Main/yuv420p/faststart', 'actual_browser_playback_checked': False}


@torch.no_grad()
def main():
    verify(); assert not (REPORT/'exports.json').exists()
    renderer = setup(); torch.set_num_threads(4); cv2.setNumThreads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    previous = ROOT/'runs/pidg_staged_reconstruction/S1_cross_temporal'
    protocol = read(REPORT/'protocol.json')
    models,states,checkpoints,fingerprints = {},{},{},{}
    for arm in ['G1_parent','control','visibility']:
        checkpoint = ROOT/protocol['parent'] if arm == 'G1_parent' else RUNS/'formal'/arm/'checkpoint.pt'
        expected = 12000 if arm == 'G1_parent' else 18000
        if arm != 'G1_parent':
            audit = read(REPORT/f'formal_{arm}_audit.json')
            assert audit['passed'] and audit['iteration'] == expected and audit['checkpoint_sha256'] == sha(checkpoint)
        state = torch.load(checkpoint,map_location='cpu',weights_only=False); assert state['iteration'] == expected
        model = AdaptiveGaussian(PREVIOUS_RUNS/'initialization.npz'); model.load_payload(state['model'])
        models[arm],states[arm],checkpoints[arm],fingerprints[arm] = model,state,checkpoint,sha(checkpoint)
    sequences = [[json.loads(line)['frame_index'] for line in (RUNS/'formal'/a/'sampling.jsonl').read_text().splitlines()] for a in ['control','visibility']]
    assert sequences[0] == sequences[1] and len(sequences[0]) == 6000
    old_report = ROOT/'experiments/staged_reconstruction/S1_cross_temporal'
    old_discovery, old_protocol = read(old_report/'discovery.json'), read(old_report/'protocol.json')
    with np.load(ROOT/'runs/pidg_staged_reconstruction/S1_automatic/inferred_masks.npz') as data: old_masks = {k: data[k] for k in data.files}
    with np.load(previous/'geometric_confidence.npz') as data: ct_masks = {k: data[k] >= .5 for k in data.files}
    with np.load(PREVIOUS_RUNS/'geometric_confidence.npz') as data: geometry = {k: data[k] for k in data.files}
    with np.load(PREVIOUS_RUNS/'reservoir.npz') as data: reservoir = {k: data[k] for k in data.files}
    selected = [2, 27, 43, 86, 166, 188, 200]
    rows, panels, images, cameras = [], [], {}, {}
    for split in ['train', 'val']:
        meta = read(DATA/'transforms_train.json' if split == 'train' else ROOT/'data/derived/kinematic_resplit/transforms_val.json')
        for frame in meta['frames']:
            number = int(Path(frame['file_path']).stem); name = str(number)
            small = rgb(frame, 400); target = torch.from_numpy(rgb(frame, 1600)).permute(2, 0, 1).cuda()
            cam = camera(frame, meta, 1600); fingerprint = camera_hash(cam)
            if split == 'train':
                masks = {'common_old_seed_train_proxy': old_masks[name], 'common_CT_train_proxy': ct_masks[name],
                         'common_new_geometry_train_proxy': geometry[name] >= .5}
            else:
                _, inferred = region_confidence(frame, small, old_discovery['anchors'], old_discovery['incompatible'], meta, old_protocol)
                masks = {'common_CT_val_inferred_not_GT': inferred >= .5,
                         'common_new_val_inferred_not_GT': new_inferred_mask(frame, small, meta, reservoir)}
            row = {'split': split, 'frame': number, 'models': {}}
            rendered = []
            for arm, model in models.items():
                package = render_static(renderer, cam, model); value = pixels(package['render'])
                row['models'][arm] = {label: metric(package, target, mask) for label, mask in masks.items()}
                assert camera_hash(cam) == fingerprint
                if split == 'train' and number in selected:
                    path = OUT/f'{arm}_t{number:04d}_native.png'; Image.fromarray(value).save(path)
                    images[path.relative_to(ROOT).as_posix()] = sha(path); rendered.append((arm+('12k' if arm == 'G1_parent' else '18k'), value))
            rows.append(row)
            if split == 'train' and number in selected:
                cameras[name] = fingerprint
                path = OUT/f'observed_t{number:04d}_native.png'; Image.fromarray(pixels(target)).save(path)
                images[path.relative_to(ROOT).as_posix()] = sha(path)
                p = panel([('Observed RGB includes motion', pixels(target))]+rendered, number)
                panels.append(np.asarray(p))
    Image.fromarray(np.concatenate(panels)).save(OUT/'native_comparison.png')
    train_meta = read(DATA/'transforms_train.json')
    fixed_videos = {}
    for view in [86, 188]:
        frame = next(f for f in train_meta['frames'] if int(Path(f['file_path']).stem) == view)
        cam = camera(frame, train_meta, 1600); fingerprint = camera_hash(cam)
        fixed_images = [(a+('12k' if a == 'G1_parent' else '18k'), pixels(render_static(renderer, cam, g)['render'])) for a, g in models.items()]
        for name, value in fixed_images:
            for time in [2, 100, 200]:
                path = OUT/f'fixed{view}_{name.split()[0]}_time{time:04d}.png'; Image.fromarray(value).save(path)
                images[path.relative_to(ROOT).as_posix()] = sha(path)
        assert camera_hash(cam) == fingerprint == cameras[str(view)]
        fixed_videos[str(view)] = encode(OUT/f'fixed{view}_comparison.mp4', (panel(fixed_images, t) for t in range(1, 201)))
    from scene_PIDG.gaussian_model import GaussianModel
    from plyfile import PlyData
    plys = {}
    for arm in ['control', 'visibility']:
        native = GaussianModel(3); native.active_sh_degree = 3
        for key, parameter in models[arm].named_parameters(): setattr(native, key, parameter)
        native.dynamic_mask = torch.zeros(len(models[arm].get_xyz), dtype=torch.bool, device='cuda')
        path = OUT/f'{arm}_18000.ply'; native.save_ply(str(path))
        vertex = PlyData.read(path)['vertex']; xyz = np.stack([vertex[k] for k in ['x', 'y', 'z']], 1)
        assert np.array_equal(xyz, models[arm].get_xyz.cpu().numpy()) and np.isfinite(xyz).all()
        plys[arm] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'vertices': len(xyz), 'checkpoint_positions_exact': True}
    all_meta = read(ROOT/'data/PIDG/rubber_duck/transforms_all.json')
    all_frames = sorted(all_meta['frames'], key=lambda f: int(Path(f['file_path'].replace('\\', '/')).stem))
    def trajectory():
        for frame in all_frames:
            number = int(Path(frame['file_path'].replace('\\', '/')).stem)
            cam = camera(frame, all_meta, 1600); fingerprint = camera_hash(cam)
            values = [('Observed RGB includes motion', (observed(frame, 400)*255).round().astype('uint8'))]
            values += [(a+('12k' if a == 'G1_parent' else '18k'), pixels(render_static(renderer, cam, g)['render'])) for a, g in models.items()]
            assert camera_hash(cam) == fingerprint
            yield panel(values, number)
    video = encode(OUT/'original_trajectory_comparison.mp4', trajectory()); assert video['decoded_frames'] == 200
    labels = sorted({label for row in rows for label in row['models']['G1_parent']})
    aggregate = {}
    for label in labels:
        aggregate[label] = {}
        for arm in models:
            values = [r['models'][arm].get(label) for r in rows if r['models'][arm].get(label) is not None]
            aggregate[label][arm] = {'nonempty_frames': len(values), 'pixels_total': sum(v['pixels'] for v in values),
                **{k: float(np.mean([v[k] for v in values])) if values else None for k in ['PSNR', 'alpha_mean', 'alpha_below_05']}}
    assert all(sha(checkpoints[a]) == fingerprints[a] for a in models); verify()
    write(REPORT/'exports.json', {'passed': True, 'stage': 'S1-C', 'revision': 1, 'iteration': 18000,
        'metrics': aggregate, 'frames': rows, 'checkpoint_sha256': fingerprints,
        'point_count': {a: len(g.get_xyz) for a, g in models.items()}, 'matched_full12k_parent_and_6000_camera_sequence': True,
        'confidence_summary': {a: summary(states[a]['confidence']) for a in ['control', 'visibility']},
        'removed_ids': states['visibility']['removed_ids'], 'fixed_topology_during_continuation': True, 'fixed_camera_fingerprints': cameras,
        'images': images, 'PLY': plys, 'original_trajectory_video': video, 'fixed_videos': fixed_videos,
        'evaluation_source_sha256': sha(Path(__file__)), 'independent_dataset_static_or_geometry_GT': False,
        'static_fixed_camera_time_invariance_is_by_construction': True, 'full_reconstruction_claimed': False,
        'validation_mask_contract': 'Frozen historical CT inferred region plus z-buffered train-reservoir 2px projections with RGB L1<=.12; common across all three models; both are proxies. Validation images never enter training.'})
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>S1-C visibility comparison</title>'
        '<style>body{background:#161922;color:white;font:17px system-ui;max-width:1300px;margin:25px auto}img,video{max-width:100%}a{color:#9cf}</style>'
        '<h1>G1 parent12k / matched control18k / visibility18k</h1><p>Partial static reconstruction. Inferred masks/depths are conditional proxies, not independent GT.</p>'
        '<p><a href="control_18000.ply">Control PLY</a> | <a href="visibility_18000.ply">Visibility PLY</a></p>'
        '<h2>Native camera comparison</h2><img src="native_comparison.png">'
        '<h2>Original 200 camera trajectory</h2><video controls preload="metadata" poster="visibility_t0086_native.png" src="original_trajectory_comparison.mp4"></video>'
        '<h2>Fixed86</h2><p>Static models are time invariant by construction. No independent fixed-camera temporal GT.</p>'
        '<video controls preload="metadata" src="fixed86_comparison.mp4"></video><h2>Fixed188</h2><video controls preload="metadata" src="fixed188_comparison.mp4"></video>',encoding='utf8')
    print('CLEAN_STATIC_EVALUATED',aggregate,flush=True)


if __name__ == '__main__': main()
