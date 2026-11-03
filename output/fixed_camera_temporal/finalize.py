"""Verify encoded artifacts and export a browseable temporal diagnostic."""
from pathlib import Path
import hashlib
import html
import json
import sys
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPORT = ROOT/'experiments/fixed_camera_temporal'
NAMES = ['C1_original_50k', 'T1_association_52k', 'O1_flow_on_50k']
VIEWS = ['fixed_0002', 'fixed_0086', 'fixed_0188', 'moving_control']
SELECTED = [2, 27, 43, 74, 86, 108, 172, 200]


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, v):
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    proof = read(REPORT/'render_verification.json')
    assert proof['optimizer_updates'] == 0
    for p, h in proof['checkpoint_sha256_unchanged'].items():
        assert digest(Path(p)) == h
    for p, h in proof['source_sha256_unchanged'].items():
        assert digest(ROOT/p) == h
    summary = {}
    parity = {}
    for name in NAMES:
        result = read(REPORT/f'{name}_results.json')
        assert result['optimizer_updates'] == 0 and result['checkpoint_unchanged']
        assert result['fixed_camera_fingerprints_before'] == result['fixed_camera_fingerprints_after']
        assert len(set(result['fixed_camera_fingerprints_before'].values())) == 3
        if name in ['C1_original_50k', 'T1_association_52k']:
            previous = ROOT/result['preflight']['reference']
            assert digest(previous) == result['preflight']['reference_sha256']
            before = np.asarray(Image.open(previous).convert('RGB'))[:, 1600:]
            after = np.asarray(Image.open(OUT/f'preflight_{name}.png').convert('RGB'))
            delta = np.abs(before.astype(int)-after.astype(int))
            parity[name] = int(delta.max())
            assert parity[name] == result['preflight']['native_diagonal_max_LSB_error'] == 0
        for view in VIEWS:
            rows = result['frames'][view]
            assert [r['frame'] for r in rows] == list(range(1, 201))
            base = OUT/name/view
            assert len(list((base/'small').glob('rgb*.png'))) == 200
            assert len(list((base/'small').glob('alpha*.png'))) == 200
            assert len(list((base/'native').glob('*.png'))) == 16
            for n in SELECTED:
                with Image.open(base/'native'/f'rgb{n:04d}.png') as native:
                    assert native.size == (1600, 900)
                    expected = np.asarray(native.resize((400, 225), Image.Resampling.LANCZOS))
                actual = np.asarray(Image.open(base/'small'/f'rgb{n:04d}.png'))
                assert np.array_equal(expected, actual)
            stats = {}
            for mask in ['alpha_05', 'alpha_09', 'yellow_appearance']:
                by_area = sorted(rows, key=lambda r: r[mask]['area'])
                stats[mask] = {
                    'area_range': [by_area[0][mask]['area'], by_area[-1][mask]['area']],
                    'area_extreme_frames': [by_area[0]['frame'], by_area[-1]['frame']],
                    'component_count_ge20px_range': [min(r[mask]['components_ge20px'] for r in rows),
                                                   max(r[mask]['components_ge20px'] for r in rows)],
                    'highest_hole_area_frames': [r['frame'] for r in sorted(rows, key=lambda r: -r[mask]['hole_area_ge4px'])[:5]],
                    'highest_small_fragment_area_frames': [r['frame'] for r in sorted(rows, key=lambda r: -r[mask]['small_component_area'])[:5]]}
            summary[f'{name}/{view}'] = stats
    write(REPORT/'descriptive_summary.json', summary)
    decoded = {}
    for path in sorted(OUT.rglob('*.mp4')):
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
        meta = next(reader)
        count = 0; chosen = []
        expected_size = (1200, 510) if 'comparison' in path.name else (800, 480)
        assert tuple(meta['size']) == expected_size
        assert abs(meta['fps']-25) < 1e-8
        for frame in reader:
            count += 1
            if 'comparison' in path.name and count in SELECTED:
                array = np.frombuffer(frame, np.uint8).reshape(meta['size'][1], meta['size'][0], 3)
                chosen.append((count, Image.fromarray(array.copy())))
        assert count == 200, (path, count)
        if chosen:
            contact = Image.new('RGB', (1200, len(chosen)*190), '#171a21')
            draw = ImageDraw.Draw(contact)
            for k, (n, im) in enumerate(chosen):
                draw.text((8, k*190+2), f'DECODED VIDEO | time frame {n}', fill='white')
                # RGB rows only, at half width, for independent codec inspection.
                contact.paste(im.crop((0, 30, 1200, 255)).resize((900, 169)), (0, k*190+20))
            contact.save(path.with_suffix('.decoded_contact.png'))
        decoded[str(path.relative_to(OUT))] = {'frames_decoded': count, 'dimensions': meta['size'], 'fps': meta['fps'], 'sha256': digest(path)}
        print('decoded', path.name, count, flush=True)
    assert len(decoded) == 16
    write(REPORT/'video_verification.json', decoded)
    # Make one concise overview for each model: rows are fixed cameras + moving control.
    for name in NAMES:
        times = [2, 43, 86, 172, 200]
        contact = Image.new('RGB', (2000, 4*252), '#171a21')
        draw = ImageDraw.Draw(contact)
        for y, view in enumerate(VIEWS):
            for x, n in enumerate(times):
                draw.text((400*x+6, 252*y+5), f'{view} | frame {n}', fill='white')
                contact.paste(Image.open(OUT/name/view/'small'/f'rgb{n:04d}.png'), (x*400, y*252+27))
        contact.save(OUT/name/'temporal_overview.png')
    # The original trajectory has real observations; the fixed view/time cross-products do not.
    key = Image.new('RGB', (1200, 1008), '#171a21')
    draw = ImageDraw.Draw(key)
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    key_inputs = {}
    for x, n in enumerate([2, 86, 200]):
        gt_path = ROOT/f'data/PIDG/rubber_duck/train/{n:04d}.png'
        key_inputs[str(gt_path.relative_to(ROOT))] = digest(gt_path)
        with Image.open(gt_path) as im:
            rgba = np.asarray(im.convert('RGBA')).astype(np.float32)/255
        gt = Image.fromarray((rgba[..., :3]*rgba[..., 3:4]*255).round().astype('uint8'))
        paths = [gt.resize((400, 225), Image.Resampling.LANCZOS),
                 Image.open(OUT/'C1_original_50k/moving_control/small'/f'rgb{n:04d}.png'),
                 Image.open(OUT/'C1_original_50k/fixed_0086/small'/f'rgb{n:04d}.png'),
                 Image.open(OUT/'C1_original_50k/fixed_0086/small'/f'alpha{n:04d}.png').convert('RGB')]
        labels = ['实际图像（原相机轨迹）', 'C1 渲染（原相机轨迹）',
                  'C1 渲染（固定第 86 帧相机）', '固定相机透明度（白色为不透明）']
        for y, (im, label) in enumerate(zip(paths, labels)):
            draw.text((400*x+8, 252*y+4), f'{label} | 时刻 {n}', fill='white', font=font)
            key.paste(im, (400*x, 252*y+27))
    key.save(OUT/'key_comparison.png')
    write(REPORT/'key_figure_GT_provenance.json', key_inputs)
    parts = ['<!doctype html><meta charset="utf-8"><title>固定相机时间序列检查</title>',
             '<style>body{background:#11151c;color:#eee;font:17px system-ui;margin:26px;max-width:1600px}video,img{max-width:100%}section{border-top:1px solid #555;margin-top:36px}a{color:#8bc9ff}button{padding:10px;margin:5px}</style>',
             '<h1>固定相机，只改变时间</h1><p>三个固定训练相机 × 200 时刻。RGB 上排、透明度下排。左 C1，中 T1，右 O1。播放速度为每秒 25 帧，与真实时间单位无关。</p>',
             '<p>模型未训练、参数未修改；1600×900 原生渲染后缩图。新相机/时刻组合没有真值，不能据此确认真实运动或材料身份。透明度为整个场景，包括蓝色结构。</p>',
             '<p><a href="../../experiments/fixed_camera_temporal/assessment.md">中文结论与限制</a></p>']
    for view in VIEWS:
        parts += [f'<section><h2>{view}</h2>',
                  f'<video controls loop preload="metadata" src="{view}_comparison_RGB_alpha.mp4"></video>',
                  f'<p><a href="{view}_comparison_RGB_alpha.decoded_contact.png">视频解码接触表</a></p>']
        for name in NAMES:
            parts += [f'<details><summary>{name}：RGB 视频、原图和接触表</summary>',
                      f'<video controls loop preload="none" src="{name}/{view}_RGB.mp4"></video>',
                      f'<img loading="lazy" src="{name}/{view}_RGB_alpha_contact.png">', '<p>原生 RGB / 透明度： ']
            for n in SELECTED:
                parts += [f'<a href="{name}/{view}/native/rgb{n:04d}.png">{n} RGB</a> / ',
                          f'<a href="{name}/{view}/native/alpha{n:04d}.png">alpha</a>　']
            parts += ['</p></details>']
        parts += ['</section>']
    (OUT/'index.html').write_text('\n'.join(parts), encoding='utf-8')
    paths = [p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    write(REPORT/'artifact_manifest.json', {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)})
    write(REPORT/'final_verification.json', {
        'checkpoint_and_source_hashes_unchanged': True, 'optimizer_updates': 0,
        'models': 3, 'fixed_camera_poses': 3, 'time_samples_per_camera': 200,
        'native_renders_including_moving_control': 2400, 'videos_decoded': 16,
        'selected_native_rgb_pngs': 96, 'selected_native_alpha_pngs': 96,
        'selected_native_to_small_RGB_exact': True,
        'fixed_camera_fingerprints_unchanged': True,
        'C1_T1_native_diagonal_parity_max_LSB': max(parity.values()),
        'novel_camera_time_GT_available': False,
        'manual_review_record': 'experiments/fixed_camera_temporal/manual_review.json',
        'files_in_manifest': len(paths), 'finalizer_sha256': digest(Path(__file__))})
    print('Final verification passed', flush=True)


if __name__ == '__main__':
    main()
