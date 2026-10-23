"""Pixel-preserving GT/render pairs for all four isotropic best snapshots."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pidg_benchmark import PROJECT, read_json, sha256, write_json, write_text_atomic

RUN = PROJECT / 'runs/pidg_isotropic_tuning/I2_iso_1e3/rubber_duck'
OUTPUT = PROJECT / 'runs/pidg_isotropic_tuning/render_comparisons/I2_iso_1e3'
REPORTS = PROJECT / 'experiments/isotropic_tuning'
METRICS = ('PSNR', 'SSIM', 'LPIPS', 'MS-SSIM')
SIZE = (1600, 900)
EXPECTED = {f'{i:05d}.png' for i in range(50)}


def rgb(path):
    with Image.open(path) as source:
        assert source.size == SIZE, (path, source.size)
        return source.convert('RGB')


def pair_frame(task):
    objective, selected, name = task
    tag = selected['snapshot']
    raw = RUN / 'test' / ('ours_' + tag)
    gt, render = rgb(raw / 'gt' / name), rgb(raw / 'renders' / name)
    reference = rgb(PROJECT / 'runs/pidg_official/rubber_duck/test/ours_best/gt' / name)
    assert gt.tobytes() == reference.tobytes(), (tag, name, 'GT differs from baseline')
    canvas = Image.new('RGB', (3200, 950), '#edf2f7')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    draw.text((18, 10), f'GT | official test {name[:-4]}', fill='#17304a', font=font)
    draw.text((1618, 10), f'Render | {objective} best | step {selected["iteration"]} | isotropic weight 0.001', fill='#17304a', font=font)
    canvas.paste(gt, (0, 50)); canvas.paste(render, (1600, 50))
    target = OUTPUT / 'pairs' / tag / name
    canvas.save(target, compress_level=3)
    # Read back every saved pair: both source panels remain pixel-identical.
    with Image.open(target) as saved:
        assert saved.crop((0, 50, 1600, 950)).tobytes() == gt.tobytes()
        assert saved.crop((1600, 50, 3200, 950)).tobytes() == render.tobytes()
    return str(target)


def main():
    state = read_json(PROJECT / 'runs/pidg_isotropic_tuning/controller.json')
    assert state['status'] == 'completed'
    checkpoint = read_json(RUN / 'checkpoints/resume.json')
    assert checkpoint['iteration'] == 50000 and checkpoint['sha256'] == sha256(RUN / 'checkpoints/resume.pt')
    selected = read_json(RUN / 'metric_best.json')
    scores, per_view = read_json(RUN / 'results.json'), read_json(RUN / 'per_view.json')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths, tasks = {}, []
    for objective, best in selected.items():
        tag = best['snapshot']; raw = RUN / 'test' / ('ours_' + tag)
        marker = read_json(raw / 'render_complete.json')
        assert marker['frames'] == 50 and marker['snapshot']
        assert all(sha256(Path(p)) == value for p, value in marker['snapshot'].items())
        assert all({p.name for p in (raw / kind).glob('*.png')} == EXPECTED for kind in ('gt', 'renders'))
        assert all(set(per_view['ours_' + tag][key]) == EXPECTED for key in METRICS)
        assert all(abs(scores['ours_' + tag][key] - best['metrics'][key]) < 1e-4 for key in METRICS)
        pairs = OUTPUT / 'pairs' / tag; pairs.mkdir(parents=True, exist_ok=True)
        paths[objective] = {'snapshot': tag, 'iteration': best['iteration'], 'metrics': scores['ours_' + tag],
                            'renders_absolute': str(raw / 'renders'), 'gt_absolute': str(raw / 'gt'),
                            'pairs_absolute': str(pairs), 'frames': 50}
        tasks.extend((objective, best, name) for name in sorted(EXPECTED))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(pair_frame, tasks), 1):
            if i % 25 == 0:
                print(f'GT/render pairs verified {i}/{len(tasks)}', flush=True)
    for row in paths.values():
        assert {p.name for p in Path(row['pairs_absolute']).glob('*.png')} == EXPECTED

    preview = Image.new('RGB', (1984, 912), '#101725')
    draw = ImageDraw.Draw(preview); font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 24)
    draw.text((16, 14), 'Rubber Duck | isotropic weight 0.001 | official test 00025', font=font, fill='#e5edf8')
    panels = [('GT', rgb(Path(paths['PSNR']['gt_absolute']) / '00025.png'))]
    panels += [(f"{key} best | step {row['iteration']}", rgb(Path(row['renders_absolute']) / '00025.png')) for key, row in paths.items()]
    for i, (label, panel) in enumerate(panels):
        x, y = 16 + i % 3 * 656, 68 + i // 3 * 416
        draw.text((x, y), label, font=font, fill='#83e1c0')
        preview.paste(panel.resize((640, 360), Image.Resampling.LANCZOS), (x, y + 36))
    preview.save(OUTPUT / 'gt_and_four_snapshots.png')
    options = ''.join(f'<option value="{html.escape(row["snapshot"])}">{html.escape(key)} best · step {row["iteration"]}</option>' for key, row in paths.items())
    rows = ''.join('<tr><td>' + '</td><td>'.join([html.escape(key), str(row['iteration']), *[f'{row["metrics"][k]:.7f}' for k in METRICS]]) + '</td></tr>' for key, row in paths.items())
    template = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Isotropic · GT / Render</title>
<style>body{max-width:1500px;margin:30px auto;padding:0 24px;font:16px/1.6 system-ui;background:#101725;color:#e5edf8}h1{font-size:26px}img{width:100%;height:auto}a{color:#83e1c0}select,button,input{font:inherit;padding:7px;margin:6px;color:inherit;background:#1d2b3e;border:1px solid #40516b;border-radius:6px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:10px;border-bottom:1px solid #344255}.controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px}input{min-width:260px}.note{color:#b4c2d6}</style>
<h1>I2 isotropic normalization · GT / Render 对照</h1>
<p>50k 训练和四组最佳快照的渲染已完成。每组50张官方测试图，左侧GT，右侧Render。</p>
<div class="controls"><label>快照 <select id="snapshot">__OPTIONS__</select></label><button id="prev">上一帧</button><label>帧 <input id="frame" type="range" min="0" max="49" value="25"></label><span id="number">00025</span><button id="next">下一帧</button><a id="full" target="_blank">打开原分辨率拼接图</a></div>
<img id="pair" alt="左侧GT，右侧当前快照的Render"><p class="note">拼接图为3200×950，包含两张1600×900原图和50像素标题；图像区域不缩放、不改像素。网页会按窗口宽度显示。SSIM与MS-SSIM最佳快照均在10k步选出。</p>
<table><thead><tr><th>快照目标</th><th>额外步数</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>MS-SSIM ↑</th></tr></thead><tbody>__ROWS__</tbody></table>
<h2>GT + 四个最佳快照总览</h2><a href="gt_and_four_snapshots.png" target="_blank"><img src="gt_and_four_snapshots.png" alt="第25张测试图的GT与四个最佳快照"></a>
<p class="note">总览图仅用于展示，已缩放；指标来自完整分辨率的原始渲染。快照按官方测试集指标选择。</p><p><a href="manifest.json">所有原始Render、GT及拼接图的绝对目录</a></p>
<script>const s=document.getElementById('snapshot'),f=document.getElementById('frame');function update(){let n=String(f.value).padStart(5,'0');document.getElementById('number').textContent=n;let p='pairs/'+s.value+'/'+n+'.png';document.getElementById('pair').src=p;document.getElementById('full').href=p;}s.value='metric_lpips';s.onchange=update;f.oninput=update;document.getElementById('prev').onclick=()=>{f.value=Math.max(0,Number(f.value)-1);update();};document.getElementById('next').onclick=()=>{f.value=Math.min(49,Number(f.value)+1);update();};update();</script></html>'''
    write_text_atomic(OUTPUT / 'index.html', template.replace('__OPTIONS__', options).replace('__ROWS__', rows))
    manifest = {'status': 'completed', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'training_completed_steps': 50000, 'checkpoint_sha256_verified': checkpoint['sha256'],
                'index_absolute': str(OUTPUT / 'index.html'), 'preview_absolute': str(OUTPUT / 'gt_and_four_snapshots.png'),
                'pair_count': len(tasks), 'pair_size': [3200, 950], 'original_panel_size': list(SIZE),
                'pair_pixel_identity_verified': True, 'gt_matches_original_baseline': True,
                'four_snapshot_weight_hashes_verified': True, 'paths': paths}
    write_json(OUTPUT / 'manifest.json', manifest)
    write_json(REPORTS / 'render_comparisons.json', manifest)
    report = REPORTS / 'results.md'
    text = report.read_text(encoding='utf-8').split('<!-- isotropic-render-pairs -->')[0].rstrip()
    write_text_atomic(report, text + '\n\n<!-- isotropic-render-pairs -->\n## GT / Render pairs\n\n[Open comparison gallery](../../runs/pidg_isotropic_tuning/render_comparisons/I2_iso_1e3/index.html) · [Absolute image paths](render_comparisons.json)\n\nFour best snapshots × 50 official test views = 200 pixel-preserving GT/render pairs.\n')
    print('Gallery: ' + str(OUTPUT / 'index.html'), flush=True)


if __name__ == '__main__':
    main()
