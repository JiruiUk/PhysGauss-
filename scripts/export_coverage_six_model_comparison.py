"""Combine existing final-50k test renders without altering source pixels."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/pidg_coverage_comparison/final50000'
SPLIT = ROOT / 'data/derived/kinematic_resplit/final_test/transforms_test.json'
ARMS = [
    ('C0_V3_reference', 'V3 reference', 'pidg_coverage_tuning'),
    ('C1_mask', 'Foreground mask', 'pidg_coverage_tuning'),
    ('C2_mask_affine', 'Mask + local affine', 'pidg_coverage_tuning'),
    ('C3_mask_affine_density', 'Mask + affine + density', 'pidg_coverage_tuning'),
    ('C4_affine_only', 'Local affine only', 'pidg_coverage_isolated'),
    ('C5_density_only', 'Coverage density only', 'pidg_coverage_isolated'),
]
WIDTH, HEIGHT, HEADER = 1600, 900, 64


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_directory(arm):
    return ROOT / 'runs' / arm[2] / arm[0] / 'rubber_duck/test/ours_50000'


def export_frame(item):
    index, frame = item
    filename = f'{index:05d}.png'
    original_id = int(Path(frame['file_path']).name)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 29)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 27)
    canvas = Image.new('RGB', (WIDTH * 4, (HEIGHT + HEADER) * 2), '#edf2f7')
    draw = ImageDraw.Draw(canvas)
    sources, tiles, expected = [], [], []
    gt_path = raw_directory(ARMS[0]) / 'gt' / filename
    with Image.open(gt_path) as image:
        gt = image.convert('RGB')
    if gt.size != (WIDTH, HEIGHT):
        raise ValueError(f'Unexpected GT dimensions: {gt_path}')
    for arm in ARMS:
        other_gt = raw_directory(arm) / 'gt' / filename
        with Image.open(other_gt) as image:
            if image.convert('RGB').tobytes() != gt.tobytes():
                raise ValueError(f'GT mismatch: {other_gt}')
    inputs = [('GT', 'Ground truth', gt_path)] + [
        (arm[0], arm[1], raw_directory(arm) / 'renders' / filename) for arm in ARMS
    ]
    for cell, (name, description, source) in enumerate(inputs):
        x, y = (cell % 4) * WIDTH, (cell // 4) * (HEIGHT + HEADER)
        with Image.open(source) as image:
            image = image.convert('RGB')
            if image.size != (WIDTH, HEIGHT):
                raise ValueError(f'Unexpected render dimensions: {source}')
            box = [x, y + HEADER, x + WIDTH, y + HEADER + HEIGHT]
            canvas.paste(image, box[:2])
            expected.append((box, image.tobytes()))
        draw.text((x + 22, y + 17), f'{name} | {description}', font=font, fill='#17304a')
        sources.append({'model': name, 'absolute_path': str(source), 'sha256': digest(source)})
        tiles.append({'model': name, 'pixel_box_xyxy': box})
    x, y = WIDTH * 3 + 60, HEIGHT + HEADER + 90
    lines = [
        'Rubber Duck | Coverage ablation',
        f'Original frame: {original_id:04d} | export: {index:05d}',
        f'Shared normalized time: {(original_id - 1) / 199:.6f}',
        'Snapshot: final iteration 50000 (all six models)',
        'Split: fixed 50-frame held-out test set',
        'Top row: GT / C0 / C1 / C2',
        'Bottom row: C3 / C4 / C5 / information',
        'Each image: original 1600 x 900 pixels',
        'No rescaling, recoloring, or crop.',
        'GT identity checked across all six runs.',
        'All seven exported tiles verified pixel-for-pixel.',
    ]
    for line in lines:
        draw.text((x, y), line, font=small, fill='#17304a')
        y += 58
    target = OUTPUT / filename
    canvas.save(target, compress_level=3)
    with Image.open(target) as check:
        for box, pixels in expected:
            if check.crop(box).tobytes() != pixels:
                raise ValueError(f'Output tile differs from source: {target}, {box}')
    return {'export_index': index, 'original_frame_id': original_id,
            'normalized_time': (original_id - 1) / 199,
            'filename': filename, 'sha256': digest(target),
            'gt_identical_across_six_runs': True,
            'all_tiles_pixel_identity_verified': True, 'tiles': tiles, 'sources': sources}


def main():
    frames = json.loads(SPLIT.read_text(encoding='utf-8'))['frames']
    ids = [int(Path(frame['file_path']).name) for frame in frames]
    if ids != list(range(1, 201, 4)):
        raise ValueError('Unexpected fixed-test frame mapping; do not infer file alignment')
    for arm in ARMS:
        for kind in ('gt', 'renders'):
            actual = {path.name for path in (raw_directory(arm) / kind).glob('*.png')}
            if actual != {f'{index:05d}.png' for index in range(50)}:
                raise ValueError(f'Incomplete or ambiguous input: {arm[0]} / {kind}')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(export_frame, enumerate(frames)))
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'output_absolute': str(OUTPUT), 'snapshot_iteration': 50000,
                'comparison_count': len(records), 'images_per_comparison': 7,
                'canvas_size': [WIDTH * 4, (HEIGHT + HEADER) * 2],
                'source_panel_size': [WIDTH, HEIGHT], 'split_absolute': str(SPLIT),
                'split_sha256': digest(SPLIT), 'models': [arm[0] for arm in ARMS],
                'all_source_pixels_preserved': True, 'frames': records}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GT + C0–C5 | final 50k</title>
<style>body{background:#101725;color:#e6edf5;font:16px system-ui;margin:24px}
h1{font-size:25px}a{color:#8ae3c0}img{width:100%;height:auto;background:#edf2f7}
.controls{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:20px 0}
input{width:300px}button{padding:8px 14px;cursor:pointer}#status{min-width:260px}</style>
<h1>GT + C0–C5：六模型拼接对比</h1>
<p>统一使用最终 50,000 iter；固定 50 张测试帧。上排：GT / C0 / C1 / C2；下排：C3 / C4 / C5 / 帧信息。</p>
<p>每个图块保留原始 1600 × 900 像素，整图 6400 × 1928；网页预览自适应宽度，点击图片可查看原图。</p>
<div class="controls"><button id="prev">上一帧</button><input id="frame" type="range" min="0" max="49" value="38">
<button id="next">下一帧</button><span id="status"></span><a id="full" target="_blank">打开原图</a></div>
<a id="imageLink" target="_blank"><img id="comparison" alt="GT and six model renders"></a>
<p>左右方向键切换。所有 GT 已核对相同；导出后七个图块逐像素核验通过。<a href="manifest.json">来源与核验记录</a></p>
<script>const frames=FRAME_RECORDS;
const slider=document.getElementById('frame');
function update(){const item=frames[Number(slider.value)];
document.getElementById('comparison').src=item.filename;
document.getElementById('imageLink').href=item.filename;
document.getElementById('full').href=item.filename;
document.getElementById('status').textContent=`第 ${item.export_index+1}/50 张 | 原始帧 ${item.original_frame_id} | t=${item.normalized_time.toFixed(6)}`;}
function step(d){slider.value=Math.max(0,Math.min(49,Number(slider.value)+d));update();}
slider.oninput=update;document.getElementById('prev').onclick=()=>step(-1);
document.getElementById('next').onclick=()=>step(1);
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();step(e.key==='ArrowLeft'?-1:1);}});
update();</script></html>'''
    lightweight = [{key: row[key] for key in ('export_index', 'original_frame_id', 'normalized_time', 'filename')} for row in records]
    (OUTPUT / 'index.html').write_text(html.replace('FRAME_RECORDS', json.dumps(lightweight)), encoding='utf-8')
    (OUTPUT / 'README.md').write_text(
        '# GT + C0–C5 final50k comparison\n\n'
        f'Absolute directory: `{OUTPUT}`\n\n'
        'Open `index.html` locally. The 50 PNG files each contain GT and all six renders.\n'
        'Top row: GT / C0 / C1 / C2. Bottom row: C3 / C4 / C5 / frame information.\n'
        'Every source panel remains 1600×900, with labels outside original pixels.\n'
        'All six GTs match; every saved panel was verified pixel-for-pixel against its source.\n'
        'Snapshot: final iteration 50000, not a validation-selected best checkpoint.\n'
        'Filename 00000.png maps to original frame 1; 00049.png maps to frame 197.\n'
        'Manifest includes original frame IDs, source paths, SHA256 hashes, and panel coordinates.\n',
        encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'count': len(records),
                      'pixel_verified': True}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
