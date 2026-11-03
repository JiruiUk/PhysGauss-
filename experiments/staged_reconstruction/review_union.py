"""Count-free scene review, native RGB and H264 fixed-camera S3 comparisons."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from variants.staged_reconstruction.prepare_data import REPORT, RUNS, DATA, read, write, sha
from variants.staged_reconstruction.train_static import setup, camera, PIPE
from variants.staged_reconstruction.stage2.train import image
from variants.staged_reconstruction.union.prepare import UNION_REPORT, UNION_RUNS, verify_sources


def rgb(tensor):
    return (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)


def panel(value, label):
    im = Image.new('RGB', (800, 480), '#161922')
    im.paste(Image.fromarray(value).resize((800, 450), Image.Resampling.LANCZOS), (0, 30))
    ImageDraw.Draw(im).text((8, 6), label, fill='white', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16))
    return np.asarray(im)


def fingerprint(cam):
    value = cam.world_view_transform.detach().cpu().contiguous().numpy().tobytes()
    value += cam.full_proj_transform.detach().cpu().contiguous().numpy().tobytes()
    value += json.dumps([cam.FoVx, cam.FoVy, cam.image_width, cam.image_height]).encode()
    return hashlib.sha256(value).hexdigest()


class Video:
    def __init__(self, path, width, height):
        assert not path.exists(), f'Preserve existing video: {path}'
        self.path = path
        self.log = path.with_suffix('.encode.log').open('wb')
        self.process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-n',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '20', '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-profile:v', 'main',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self.log, creationflags=subprocess.CREATE_NO_WINDOW)

    def add(self, value):
        self.process.stdin.write(np.ascontiguousarray(value).tobytes())

    def close(self):
        self.process.stdin.close()
        code = self.process.wait(timeout=60)
        self.log.close()
        assert code == 0


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['warmup', 'physics'], required=True)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--standard-quality', action='store_true')
    args = parser.parse_args()
    verify_sources()
    torch.set_num_threads(4)
    checkpoint = UNION_RUNS/args.phase/'formal/checkpoint.pt'
    before = sha(checkpoint)
    assert before == read(checkpoint.with_name('checkpoint.json'))['sha256']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    protocol = read(UNION_REPORT/'protocol.json')
    assert state['iteration'] == protocol[args.phase+'_updates'] and state['phase'] == args.phase
    assert state['protocol_sha256'] == sha(UNION_REPORT/'protocol.json')
    renderer = setup()
    from variants.staged_reconstruction.union.model import UnionScene
    scene = UnionScene(protocol)
    scene.load_state_dict(state['model'])
    original_static = torch.load(RUNS/'S1_static/checkpoint.pt', map_location='cpu', weights_only=False)['model']
    assert all(torch.equal(v, scene.static.state_dict()[k].cpu()) for k, v in original_static.items())
    out = ROOT/'output/staged_reconstruction/S3_union'/args.phase
    out.mkdir(parents=True, exist_ok=True)
    meta = read(DATA/'transforms_train.json')
    byid = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    panels = []
    image_files = []
    for number in [2, 27, 42, 43, 52, 86, 166, 188, 200]:
        frame = byid[number]
        cam = camera(frame, meta, 1600)
        package = scene.render(renderer, cam, frame['time'])
        rgba = image(frame, 1600)
        gt = (rgba[..., :3]*rgba[..., 3:4]*255).round().astype(np.uint8)
        dynamic = renderer(cam, scene.dynamic_at(frame['time']), PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
        prediction = rgb(package['render'])
        path = out/f'train_{number:04d}_native.png'
        Image.fromarray(prediction).save(path)
        image_files.append(path)
        panels.append(np.concatenate([panel(gt, f'{number}: observed RGB'),
            panel(prediction, f'S3-U {args.phase} {state["iteration"]}'), panel(rgb(dynamic['render']), 'Unified dynamic population; no physical IDs')], axis=1))
    sheet = out/'observed_review.png'
    Image.fromarray(np.concatenate(panels, axis=0)).save(sheet)
    image_files.append(sheet)
    all_meta = read(ROOT/'data/PIDG/rubber_duck/transforms_all.json')
    frames = sorted(all_meta['frames'], key=lambda f: int(Path(f['file_path'].replace('\\', '/')).stem))
    assert len(frames) == 200
    video_rows = []
    camera_records = {}
    legacy = read(REPORT/'completion_integrity.json')
    protected_videos = {(row['stage'], row['view']): row['sha256'] for row in legacy['videos_verified']}
    for view in ['fixed86', 'fixed188', 'original']:
        writer = pair_writer = reference = None
        if not args.preview_only:
            writer = Video(out/f'{view}.mp4', 800, 450)
            pair_writer = Video(out/f'{view}_S3_vs_union.mp4', 1600, 480)
            source = ROOT/'output/staged_reconstruction/S3'/f'{view}.mp4'
            assert sha(source) == protected_videos['S3', view]
            reference = cv2.VideoCapture(str(source))
            assert reference.isOpened()
        snapshots = []
        try:
            for observed in frames:
                number = int(Path(observed['file_path'].replace('\\', '/')).stem)
                if args.preview_only and number not in [2, 86, 188]:
                    continue
                pose = byid[86 if view == 'fixed86' else 188] if view != 'original' else observed
                cam = camera(pose, meta, 1600)
                before_cam = fingerprint(cam)
                if view != 'original':
                    assert camera_records.setdefault(view, before_cam) == before_cam
                value = rgb(scene.render(renderer, cam, (number-1)/199)['render'])
                assert fingerprint(cam) == before_cam
                if number in [2, 86, 188]:
                    path = out/f'{view}_t{number:04d}_native.png'
                    Image.fromarray(value).save(path)
                    image_files.append(path)
                    snapshots.append(panel(value, f'{view}; frame {number}; S3-U {args.phase}'))
                if writer:
                    small = cv2.resize(value, (800, 450), interpolation=cv2.INTER_AREA)
                    ok, old = reference.read()
                    assert ok
                    writer.add(small)
                    pair = np.concatenate([panel(cv2.cvtColor(old, cv2.COLOR_BGR2RGB), f'Original S3 fixed groups; frame {number}'),
                        panel(small, f'S3-U unified population; frame {number}')], axis=1)
                    pair_writer.add(pair)
                    if number in [2, 86, 188]:
                        path = out/f'{view}_pair_t{number:04d}.png'
                        Image.fromarray(pair).save(path)
                        image_files.append(path)
        finally:
            if writer:
                writer.close()
                pair_writer.close()
                reference.release()
        path = out/f'{view}_review.png'
        Image.fromarray(np.concatenate(snapshots, axis=0)).save(path)
        image_files.append(path)
        if writer:
            assert sha(source) == protected_videos['S3', view]
            for item in [writer.path, pair_writer.path]:
                cap = cv2.VideoCapture(str(item))
                count = 0
                while cap.read()[0]:
                    count += 1
                assert count == 200 and cap.get(cv2.CAP_PROP_FPS) == 20.
                cap.release()
                video_rows.append({'path': item.relative_to(ROOT).as_posix(), 'sha256': sha(item), 'frames': count,
                    'fps': 20, 'codec': 'H264 Main/yuv420p/faststart', 'native_render_size': [1600, 900]})
        print('UNION_REVIEW_VIEW', args.phase, view, flush=True)
    if args.standard_quality:
        from variants.staged_reconstruction.union.evaluate import evaluate
        for split in ['val', 'test']:
            evaluate(renderer, scene, args.phase, state['iteration'], split, standard=True)
    diagnostics = []
    if args.phase == 'physics':
        for number in [2, 86, 188]:
            try:
                with torch.enable_grad():
                    terms, rows = scene.physics.loss(scene, (number-1)/199, 9000,
                        protocol['physics_samples_global'], protocol['time_step'])
                diagnostics.append({'frame': number, 'valid': True,
                    'terms': {k: float(v.detach()) for k, v in terms.items()}, 'population': rows})
            except ValueError as error:
                diagnostics.append({'frame': number, 'valid': False, 'error': str(error), 'not_zero_residual': True})
    assert sha(checkpoint) == before
    receipt = {'checkpoint_sha256': before, 'phase': args.phase, 'iteration': state['iteration'],
        'static_exactly_matches_S1': True, 'physical_body_number': None, 'manual_image_review_pending': True,
        'images': {p.relative_to(ROOT).as_posix(): sha(p) for p in image_files}, 'videos': video_rows,
        'fixed_camera_fingerprints': camera_records, 'camera_matrices_unchanged_after_render': True,
        'camera_contract': 'metadata_cv; same frozen camera function and metadata as original S3 export',
        'training_metadata_sha256': sha(DATA/'transforms_train.json'),
        'all_camera_metadata_sha256': sha(ROOT/'data/PIDG/rubber_duck/transforms_all.json'),
        'fixed_frame_internal_physics': diagnostics,
        'limits': 'Union population, no instance/number estimate; fixed-time novel views lack independent GT; not matched-capacity causal ablation'}
    write(UNION_REPORT/f'{args.phase}_{"preview" if args.preview_only else "exports"}.json', receipt)
    if video_rows:
        parts = ['<!doctype html><meta charset="utf-8"><title>S3 与 S3-U</title>',
            '<style>body{font:17px system-ui;max-width:1600px;margin:30px auto;background:#161922;color:#eee}video,img{max-width:100%}a{color:#9cf}</style>',
            '<h1>S3 与 S3-U：固定分组和统一动态池</h1><p>左：原 S3；右：S3-U。S3-U不输入也不估计物体数量，不使用物体ID。均为16k动态预热+6k S3训练，静态支撑共享S1；初始化和共享网络结构不同，不能作为单开关因果消融。</p>',
            '<p>原生1600×900，展示200帧/20fps，时间按(frame-1)/199同步；没有同步新视角几何真值。</p>']
        for view in ['fixed86', 'fixed188', 'original']:
            parts += [f'<h2>{view}</h2><video controls preload="metadata" poster="{view}_pair_t0086.png"><source src="{view}_S3_vs_union.mp4" type="video/mp4"></video>',
                f'<p><a href="{view}_S3_vs_union.mp4">直接打开并排视频</a> · <a href="{view}.mp4">S3-U单独视频</a></p>']
        (out/'index.html').write_text('\n'.join(parts)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
