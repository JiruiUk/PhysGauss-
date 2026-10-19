"""Fresh official PIDG renders and lossless GT/render comparison panels."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import html
import importlib
import json
import os
from pathlib import Path
import random
import re
import sys

from pidg_benchmark import PROJECT, completed_metrics, dataset_hashes, read_json, runtime_hashes, sha256, write_json

MODELS = ('A1', 'A2a', 'A2b', 'A4_single', 'A6_single')
REPO = PROJECT / 'variants/constitutive_pidg/code'
DATA = PROJECT / 'data/PIDG/rubber_duck'
RUNS = PROJECT / 'runs/pidg_constitutive'


def inputs():
    sources = runtime_hashes(REPO)
    data = dataset_hashes('rubber_duck')
    result = {}
    for name in MODELS:
        run = RUNS / f'm10_long_{name}/rubber_duck'
        info, _ = completed_metrics(run)
        assert (info['scene'], info['iterations'], info['resolution_width'], info['seed']) == ('rubber_duck', 50000, 1600, 0)
        assert info['source_hashes'] == sources and info['dataset_hashes'] == data
        files = [run/'cfg_args', run/'reproduction.json', run/'results.json']
        for folder in ('point_cloud/iteration_best', 'deform/iteration_best'):
            files.extend(p for p in (run/folder).rglob('*') if p.is_file())
        result[name] = {str(p.relative_to(PROJECT)): sha256(p) for p in files}
    return {'model_files': result, 'render_sources': sources, 'dataset': data}


def font(size):
    from PIL import ImageFont
    return ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', size)


@contextmanager
def video(path, size, fps):
    import imageio_ffmpeg
    writer = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec='libx264',
        quality=8, macro_block_size=1, ffmpeg_timeout=60,
        output_params=['-preset', 'fast', '-movflags', '+faststart'])
    writer.send(None)
    try:
        yield writer
    finally:
        writer.close()


def pair_frames(directory, name, best_step, views, fps):
    import numpy as np
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageChops
    raw = directory/'test/ours_best'
    names = sorted(p.name for p in (raw/'renders').glob('*.png'))
    assert len(names) == len(views) == 50
    assert names == sorted(p.name for p in (raw/'gt').glob('*.png'))
    paired = directory/'paired'; paired.mkdir()
    width, height = Image.open(raw/'renders'/names[0]).size
    canvas_size = (2*width+48, height+108)
    movie = directory/'gt_render.mp4'
    gt_hashes = []
    old_max_delta = 0
    with video(movie, canvas_size, fps) as writer:
        for i, filename in enumerate(names):
            with Image.open(raw/'gt'/filename) as source: gt = source.convert('RGB')
            with Image.open(raw/'renders'/filename) as source: pred = source.convert('RGB')
            assert gt.size == pred.size == (width, height)
            import hashlib
            gt_hashes.append(hashlib.sha256(gt.tobytes()).hexdigest())
            canvas = Image.new('RGB', canvas_size, '#101725')
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 12), 'GT', font=font(32), fill='#83e1c0')
            draw.text((width+32, 12), f'Render | {name}', font=font(32), fill='#e5edf8')
            canvas.paste(gt, (16, 64)); canvas.paste(pred, (width+32, 64))
            assert ImageChops.difference(canvas.crop((16, 64, width+16, height+64)), gt).getbbox() is None
            assert ImageChops.difference(canvas.crop((width+32, 64, 2*width+32, height+64)), pred).getbbox() is None
            caption = f'Rubber Duck | test {i:02d}/49 | {views[i]["image_name"]} | t={views[i]["time"]:.6f} | best snapshot step {best_step}'
            draw.text((16, height+75), caption, font=font(20), fill='#a8b8cc')
            canvas.save(paired/filename)
            writer.send(np.asarray(canvas))
            original = RUNS/f'm10_long_{name}/rubber_duck/test/ours_best/renders'/filename
            with Image.open(original) as old:
                delta = abs(np.asarray(pred).astype('int16')-np.asarray(old.convert('RGB')).astype('int16')).max()
                old_max_delta = max(old_max_delta, int(delta))
            if i % 10 == 0: print(f'PAIR {name}: {i+1}/50', flush=True)
    count, seconds = imageio_ffmpeg.count_frames_and_secs(movie)
    assert count == 50
    return {'name': name, 'best_snapshot_iteration': best_step, 'frames': 50,
            'raw_size': [width, height], 'paired_size': list(canvas_size),
            'video_frames_decoded': count, 'video_seconds': seconds,
            'video': str(movie), 'paired': str(paired), 'gt_pixel_hashes': gt_hashes,
            'paired_regions_pixel_exact': True, 'max_pixel_delta_vs_previous_render': old_max_delta}


def overview(output, rows, views, fps):
    import numpy as np
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    target = output/'all_models';target.mkdir()
    titles = ('GT', *MODELS)
    with video(output/'gt_and_five_models.mp4', (1984, 912), fps) as writer:
        for i in range(50):
            name = f'{i:05d}.png'
            canvas = Image.new('RGB', (1984, 912), '#101725')
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 15), f'Rubber Duck | GT + five models | test {i:02d}/49 | t={views[i]["time"]:.6f}', font=font(27), fill='#e5edf8')
            for k, title in enumerate(titles):
                x = 16+(k%3)*656; y = 80+(k//3)*416
                path = output/MODELS[0]/'test/ours_best/gt'/name if k == 0 else output/title/'test/ours_best/renders'/name
                with Image.open(path) as source: im = source.convert('RGB').resize((640,360), Image.Resampling.LANCZOS)
                draw.text((x, y), title, font=font(25), fill='#83e1c0' if k == 0 else '#e5edf8')
                canvas.paste(im, (x, y+40))
            canvas.save(target/name);writer.send(np.asarray(canvas))
    count, seconds = imageio_ffmpeg.count_frames_and_secs(output/'gt_and_five_models.mp4')
    assert count == 50
    return {'frames': count, 'seconds': seconds, 'size': [1984, 912]}


def gallery(output, rows, fps):
    cards = []
    for row in rows:
        name = row['name']
        cards.append(f'<section><h2>{html.escape(name)} <small>best step {row["best_snapshot_iteration"]}</small></h2><video controls preload="metadata" poster="{name}/paired/00000.png" src="{name}/gt_render.mp4"></video><p><a href="{name}/paired/">50 paired PNGs</a> · <a href="{name}/test/ours_best/renders/">Raw renders</a> · <a href="{name}/test/ours_best/gt/">GT</a></p></section>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rubber Duck | GT + Render</title><style>body{{margin:28px auto;max-width:1400px;padding:0 22px;background:#101725;color:#e5edf8;font:16px/1.65 "Segoe UI",sans-serif}}h1{{font-size:27px}}h2{{font-size:20px}}small,p{{color:#a8b8cc}}a{{color:#83e1c0}}video,img{{width:100%;background:black;border-radius:8px}}section{{border:1px solid #344255;padding:18px;margin:22px 0;border-radius:12px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}}.cards section{{margin:0}}@media(max-width:600px){{.cards{{display:block}}}}</style><h1>Rubber Duck — GT / Render comparison</h1><p>五个模型均重新渲染 50 个测试视角，使用结果表对应的 ours_best 快照。各模型视频左 GT、右 Render；PNG 保留原始像素，视频用于浏览。播放 {fps} fps，按测试帧索引排序，不代表物理时间速度。</p><section><h2>GT + 五模型同步总览</h2><video controls preload="metadata" poster="all_models/00000.png" src="gt_and_five_models.mp4"></video><p><a href="all_models/">50 张总览图</a> · <a href="manifest.json">渲染与校验记录</a></p></section><div class="cards">{''.join(cards)}</div></html>'''
    (output/'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=10)
    options = parser.parse_args()
    output = options.output.resolve();output.mkdir(parents=True, exist_ok=False)
    before = inputs()
    manifest = {'status': 'running', 'output': str(output), 'snapshot': 'ours_best', 'models': [],
                'fps': options.fps, 'scope': '50 held-out test views; 1600-pixel width; no training',
                'started_utc': datetime.now(timezone.utc).isoformat(), 'input_hashes': before,
                'script_sha256': sha256(Path(__file__))}
    def record():
        write_json(output/'manifest.json', manifest)
        write_json(PROJECT/'experiments/render_comparisons.json', manifest)
    record()
    try:
        os.environ['PIDG_HASH_SOURCE_DIR'] = str(PROJECT/'third_party/PIDG/hashencoder')
        os.environ['PIDG_HASH_BUILD_DIR'] = str(PROJECT/'third_party/PIDG/tmp_build')
        sys.path.insert(0, str(REPO));os.chdir(REPO)
        import numpy as np
        import torch
        from plyfile import PlyData
        official = importlib.import_module('render')
        assert Path(official.__file__).resolve() == REPO/'render.py'
        p = argparse.ArgumentParser();model = official.ModelParams(p, sentinel=True);pipeline = official.PipelineParams(p)
        p.add_argument('--conf');p.add_argument('--iteration', default='best');p.add_argument('--skip_train', action='store_true')
        first_run = RUNS/f'm10_long_{MODELS[0]}/rubber_duck'
        sys.argv = ['render.py', '-s', str(DATA), '-m', str(first_run), '--conf', str(REPO/'arguments/pidg/rubber_dack.py'), '--resolution', '1600', '--load2gpu_on_the_fly', '--iteration', 'best', '--skip_train']
        args = official.get_combined_args(p);args = official.merge_config(args, args.conf)
        args.data_device = 'cuda:0' if args.data_device == 'cuda' else args.data_device
        official.safe_state(False);random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.set_device(args.data_device)
        dataset = model.extract(args);pipe = pipeline.extract(args)
        canonical_count = len(PlyData.read(DATA/'points3d.ply').elements[0].data)
        g = official.GaussianModel(dataset.sh_degree)
        scene = official.Scene(dataset, g, load_iteration='best', shuffle=False)
        cameras = scene.getTestCameras()
        assert len(cameras) == 50
        views = [{'index': i, 'image_name': v.image_name, 'time': float(v.fid.item()), 'width': v.image_width, 'height': v.image_height} for i,v in enumerate(cameras)]
        manifest['views'] = views;record()
        background = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], dtype=torch.float32, device='cuda')
        with torch.no_grad():
            for index, name in enumerate(MODELS):
                print('MODEL_START', name, flush=True)
                random.seed(0);np.random.seed(0);torch.manual_seed(0)
                run = RUNS/f'm10_long_{name}/rubber_duck'
                if index:
                    g = official.GaussianModel(dataset.sh_degree)
                    g.load_ply(str(run/'point_cloud/iteration_best/point_cloud.ply'), og_number_points=canonical_count)
                deform = official.DeformModel(dataset.grid_args, dataset.network_args, scale_xyz=dataset.scale_xyz, reg_spatial_able=False, reg_temporal_able=False)
                deform.load_weights(str(run), iteration='best')
                material = torch.load(run/'point_cloud/iteration_best/material.pt', map_location='cuda', weights_only=True)
                g.velocity_net.load_state_dict(material['velocity_net']);g.particle_indices=material['particle_indices'];g.force_sigma.data.copy_(material['force_sigma'])
                best = int(material['iteration'])
                assert int(re.findall(r'\d+', (run/'point_cloud/iteration_best/iter.txt').read_text())[-1]) == best
                destination = output/name
                official.render_set(str(destination), dataset.load2gpu_on_the_fly, 'test', 'best', cameras, g, pipe, background, deform)
                torch.cuda.synchronize()
                row = pair_frames(destination, name, best, views, options.fps)
                if manifest['models']:assert row['gt_pixel_hashes'] == manifest['models'][0]['gt_pixel_hashes']
                manifest['models'].append(row);record()
                print('MODEL_COMPLETE', name, flush=True)
                del deform, material
                if index:g=None
                gc.collect();torch.cuda.empty_cache()
        manifest['overview'] = overview(output, manifest['models'], views, options.fps)
        gallery(output, manifest['models'], options.fps)
        assert inputs() == before, 'Original model, metric, source or dataset inputs changed'
        manifest.update(status='completed', original_inputs_unchanged=True, finished_utc=datetime.now(timezone.utc).isoformat());record()
        lines = ['# Rubber Duck GT / Render 对照', '', f'[打开可播放总览]({(output/"index.html").as_posix()})', '',
                 f'[五模型同步视频]({(output/"gt_and_five_models.mp4").as_posix()})', '',
                 '五个模型均重新渲染全部 50 个测试视角，使用与指标一致的 ours_best；左 GT、右 Render。',
                 '并排 PNG 原像素保留；视频播放 10 fps，测试索引顺序，不代表物理时间速度。', '',
                 '| 模型 | 视频 | 并排 PNG | best 步数 |', '|---|---|---|---:|']
        for row in manifest['models']:
            lines.append(f'| {row["name"]} | [播放]({Path(row["video"]).as_posix()}) | [50帧]({Path(row["paired"]).as_posix()}) | {row["best_snapshot_iteration"]} |')
        lines += ['', f'[校验记录]({(output/"manifest.json").as_posix()})', '', '每个视频均完成解码帧数检查；所有 PNG 的 GT/Render 区域像素与原帧一致，五模型的 GT 逐帧一致。原模型、数据、既有指标与源码哈希保持不变。']
        (PROJECT/'experiments/render_comparisons.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('ALL_COMPARISONS_COMPLETE', str(output/'index.html'), flush=True)
    except Exception as error:
        manifest.update(status='failed', error=repr(error));record();raise


if __name__ == '__main__':main()
