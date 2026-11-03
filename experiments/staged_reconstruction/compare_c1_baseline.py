"""Display-only synchronized C1_repaired/staged videos; no model writes or updates."""
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'experiments/staged_reconstruction'
OUT = ROOT / 'output/staged_reconstruction/baseline_comparison'
NAMES = ['C1_repaired', 'S2', 'S3', 'S4', 'S5', 'S6']
VIEWS = {'fixed86': 'fixed_0086', 'fixed188': 'fixed_0188', 'original': 'moving_control'}
SELECTED = [2, 86, 188]


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Video:
    def __init__(self, file, size):
        assert not file.exists(), f'Preserve existing output: {file}'
        self.file = file
        self.log = file.with_suffix('.encode.log').open('wb')
        self.proc = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-n',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}',
            '-r', '20', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'medium',
            '-crf', '18', '-profile:v', 'main', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(file)], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self.log,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def add(self, rgb):
        self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self):
        self.proc.stdin.close()
        code = self.proc.wait(timeout=60)
        self.log.close()
        assert code == 0, self.file


def inspect(file, expected):
    cap = cv2.VideoCapture(str(file))
    assert cap.isOpened()
    size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    assert size == list(expected) and count == 200 and fps == 20.0, (file, size, count, fps)
    return {'path': file.relative_to(ROOT).as_posix(), 'size': size, 'frames': count,
            'display_fps': fps, 'sha256': sha(file), 'codec': 'H.264 Main/yuv420p/faststart'}


def labeled(rgb, name, view, number, width, height, header, font):
    panel = Image.new('RGB', (width, height + header), '#171a21')
    picture = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    panel.paste(Image.fromarray(picture), (0, header))
    camera = 'legacy camera' if name == 'C1_repaired' else 'metadata CV camera'
    text = f'{name} | {camera}'
    draw = ImageDraw.Draw(panel)
    draw.text((8, 3), text, fill='white', font=font)
    if header >= 40:
        draw.text((8, 21), f'{view} | frame {number:03d}/200 | normalized time {(number-1)/199:.4f}', fill='#c6cbd6', font=font)
    return panel


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = REPORT / 'C1_baseline_comparison.json'
    assert not result.exists(), 'Completed comparison exists; verify it instead of overwriting'
    old = read(ROOT / 'experiments/fixed_camera_temporal/video_verification.json')
    old = {k.replace('\\', '/'): v for k, v in old.items()}
    baseline = read(ROOT / 'experiments/fixed_camera_temporal/C1_original_50k_results.json')
    assert baseline['fixed_camera_fingerprints_before'] == baseline['fixed_camera_fingerprints_after']
    source_root = ROOT / 'output/fixed_camera_temporal/C1_original_50k'
    c1_checkpoint = ROOT / 'runs/pidg_color_repair/C1_repaired/rubber_duck/checkpoints/resume.pt'
    assert sha(c1_checkpoint) == baseline['checkpoint']['sha256']
    integrity = read(REPORT / 'completion_integrity.json')
    staged_videos = {(v['stage'], v['view']): v for v in integrity['videos_verified']}
    protected = {c1_checkpoint: sha(c1_checkpoint)}
    for row in integrity['final_checkpoints']:
        file = ROOT / row['path']
        assert sha(file) == row['sha256']
        protected[file] = row['sha256']
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    records = []
    sources = []
    for view, legacy_view in VIEWS.items():
        paths = {'C1_repaired': source_root / f'{legacy_view}_RGB.mp4'}
        expected = old[f'C1_original_50k/{legacy_view}_RGB.mp4']
        assert expected['frames_decoded'] == 200 and expected['fps'] == 25.0
        assert sha(paths['C1_repaired']) == expected['sha256']
        for stage in NAMES[1:]:
            folder = 'S2_geometry_recovery' if stage == 'S2' else stage
            paths[stage] = ROOT / 'output/staged_reconstruction' / folder / f'{view}.mp4'
            assert sha(paths[stage]) == staged_videos[stage, view]['sha256']
        captures = {name: cv2.VideoCapture(str(file)) for name, file in paths.items()}
        assert all(cap.isOpened() for cap in captures.values())
        for name, cap in captures.items():
            expected_size = [800, 480] if name == 'C1_repaired' else [800, 450]
            assert [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))] == expected_size
            assert cap.get(cv2.CAP_PROP_FPS) == (25.0 if name == 'C1_repaired' else 20.0)
        writers = {
            'baseline': Video(OUT / f'{view}_C1_repaired.mp4', (800, 450)),
            'pair': Video(OUT / f'{view}_C1_vs_S6.mp4', (1600, 490)),
            'grid': Video(OUT / f'{view}_all_stages.mp4', (1200, 498))}
        snapshots = []
        try:
            for n in range(1, 201):
                frames = {}
                for name, cap in captures.items():
                    ok, bgr = cap.read()
                    assert ok, (name, n)
                    # Existing C1 export has a 30px label banner above 800x450 RGB.
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frames[name] = rgb[30:] if name == 'C1_repaired' else rgb
                    assert frames[name].shape == (450, 800, 3)
                writers['baseline'].add(frames['C1_repaired'])
                pair = Image.new('RGB', (1600, 490))
                for k, name in enumerate(['C1_repaired', 'S6']):
                    pair.paste(labeled(frames[name], name, view, n, 800, 450, 40, font), (k*800, 0))
                grid = Image.new('RGB', (1200, 498))
                for k, name in enumerate(NAMES):
                    grid.paste(labeled(frames[name], name, view, n, 400, 225, 24, font), ((k%3)*400, (k//3)*249))
                writers['pair'].add(np.asarray(pair))
                writers['grid'].add(np.asarray(grid))
                if n in SELECTED:
                    pair.save(OUT/f'{view}_pair_t{n:04d}.png')
                    grid.save(OUT/f'{view}_grid_t{n:04d}.png')
                    snapshots.append(pair)
            for name, cap in captures.items():
                assert not cap.read()[0], f'Extra source frames: {name}'
        finally:
            for cap in captures.values(): cap.release()
            for writer in writers.values(): writer.close()
        contact = Image.new('RGB', (1600, 1470))
        for k, panel in enumerate(snapshots): contact.paste(panel, (0, k*490))
        contact.save(OUT/f'{view}_pair_review.png')
        for file in paths.values():
            sources.append({'path': file.relative_to(ROOT).as_posix(), 'sha256': sha(file)})
        for kind, writer in writers.items():
            records.append({'view': view, 'kind': kind, **inspect(writer.file, {'baseline':(800,450),'pair':(1600,490),'grid':(1200,498)}[kind])})
        print('BASELINE_COMPARISON_VERIFIED', view, '3videos x200frames', flush=True)
    for file, expected in protected.items(): assert sha(file) == expected
    for item in sources: assert sha(ROOT/item['path']) == item['sha256']
    receipt = {'baseline': 'C1_repaired50k; historical export folder C1_original_50k is this repaired model, not official PIDG',
               'checkpoint_sha256': baseline['checkpoint']['sha256'], 'optimizer_updates': 0,
               'fixed_pose_frame_ids': [86,188], 'time_policy': '(frame-1)/199 for both branches; same200 frame order',
               'display_policy': 'All200 source frames retained once; C1 display25fps retimed to20fps without interpolation',
               'native_render_resolution': [1600,900], 'baseline_display_crop': [0,30,800,480],
               'comparison_scope': 'C1 legacy camera, staged metadata CV camera; different capacity/training budget. Visual context only, not matched causal ablation or novel-view GT.',
               'S1_video_omitted_reason': 'S1 trains static support only; complete dynamic scene comparison starts at S2.',
               'sources': sources, 'protected_checkpoints': {f.relative_to(ROOT).as_posix(): h for f,h in protected.items()}, 'videos': records}
    result.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    parts = ['<!doctype html><meta charset="utf-8"><title>C1_repaired 与 S2–S6 固定视角对比</title>',
             '<style>body{font:17px system-ui;max-width:1600px;margin:30px auto;padding:20px;background:#12151c;color:#eee}video,img{max-width:100%}a{color:#9cf}section{padding:20px 0;border-top:1px solid #555}summary{cursor:pointer;padding:12px 0}</style>',
             '<h1>C1_repaired 与 S2–S6 对比</h1><p>并排视频：左侧 C1_repaired（50k），右侧 S6。总览排列：C1、S2、S3 / S4、S5、S6。</p>',
             '<p>固定86/188采用相同帧号的相机，各模型保留训练时的相机约定：C1 为 legacy，本轮为 metadata CV。相机、模型容量和预算不同，因此这是诊断展示，不能作为严格消融。固定视角跨时间没有同步图像真值。</p>',
             '<p>每段200帧、20展示fps（10秒），按帧号同步；展示速度不代表真实采集秒数。S1仅重建静态支撑，因此动态场景总览从S2开始。</p>',
             '<p><a href="../index.html">返回本轮完整合集</a></p>']
    for view, title in [('fixed86','固定86视角'),('fixed188','固定188视角'),('original','原相机轨迹控制')]:
        parts += [f'<section><h2>{title}：C1 对 S6</h2><video controls playsinline preload="metadata" poster="{view}_pair_t0086.png"><source src="{view}_C1_vs_S6.mp4" type="video/mp4"></video>',
                  f'<p><a href="{view}_C1_vs_S6.mp4">直接打开并排视频</a> · <a href="{view}_pair_review.png">早/中/晚对照图</a></p>',
                  f'<details><summary>C1、S2–S6 六模型总览</summary><video controls playsinline preload="none"><source src="{view}_all_stages.mp4" type="video/mp4"></video></details>',
                  f'<details><summary>C1_repaired 单独视频</summary><video controls playsinline preload="none"><source src="{view}_C1_repaired.mp4" type="video/mp4"></video></details></section>']
    (OUT/'index.html').write_text('\n'.join(parts)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
