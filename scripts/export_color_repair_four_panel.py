"""Concatenate existing final50k test GT/C1/C2/C3 without changing pixels."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/pidg_color_repair/render_exports/final50000'
OUTPUT = SOURCE / 'gt_c1_c2_c3'
ARMS = ('C1_repaired', 'C2_repaired', 'C3_repaired')
WIDTH, HEIGHT, HEADER = 1600, 900, 50


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_frame(index, records):
    filename = f'{index:05d}.png'
    original_id = records[0]['original_frame_id']
    gt = SOURCE / 'gt' / filename
    gt_hash = digest(gt)
    inputs = [('GT', gt, gt_hash)]
    for arm, record in zip(ARMS, records):
        if record['filename'] != filename or record['original_frame_id'] != original_id:
            raise ValueError('Model exports do not refer to the same frame')
        if record['gt_sha256'] != gt_hash:
            raise ValueError('Ground truth differs between model exports')
        inputs.append((arm, SOURCE / arm / 'renders' / filename, record['render_sha256']))
    canvas = Image.new('RGB', (WIDTH * 4, HEIGHT + HEADER), '#edf2f7')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 26)
    expected, sources = [], []
    for column, (name, path, checksum) in enumerate(inputs):
        if digest(path) != checksum:
            raise ValueError(f'Source PNG no longer matches completed export: {path}')
        with Image.open(path) as image:
            image = image.convert('RGB')
            if image.size != (WIDTH, HEIGHT):
                raise ValueError(f'Unexpected source dimensions: {path}')
            box = (column * WIDTH, HEADER, (column + 1) * WIDTH, HEADER + HEIGHT)
            canvas.paste(image, box[:2])
            expected.append((box, image.tobytes()))
        label = f'{name} | frame {original_id:04d}'
        if column:
            label += ' | final50000'
        draw.text((column * WIDTH + 18, 9), label, font=font, fill='#17304a')
        sources.append({'panel': name, 'source_absolute': str(path), 'sha256': checksum,
                        'pixel_box_xyxy': list(box)})
    target = OUTPUT / filename
    temporary = target.with_name(target.stem + '.tmp.png')
    canvas.save(temporary, compress_level=3)
    with Image.open(temporary) as image:
        if any(image.crop(box).tobytes() != pixels for box, pixels in expected):
            raise ValueError('Concatenation changed original source pixels')
    temporary.replace(target)
    return {'export_index': index, 'original_frame_id': original_id,
            'normalized_time': (original_id - 1) / 199, 'filename': filename,
            'sha256': digest(target), 'all_four_panels_pixel_verified': True, 'sources': sources}


def main():
    manifests = [read(SOURCE / arm / 'manifest.json') for arm in ARMS]
    for arm, manifest in zip(ARMS, manifests):
        if manifest['arm'] != arm or manifest['iteration'] != 50000 or \
           manifest['split'] != 'test' or manifest['images'] != 50:
            raise ValueError('Only completed final50k fixed-test exports may be combined')
        if [f['original_frame_id'] for f in manifest['frames']] != list(range(1, 201, 4)):
            raise ValueError('Unexpected test frame mapping')
    if len({manifest['split_sha256'] for manifest in manifests}) != 1:
        raise ValueError('Models use different test sets')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda i: export_frame(i, [m['frames'][i] for m in manifests]), range(50)))
    manifest = {'status': 'completed', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'output_absolute': str(OUTPUT), 'gallery_absolute': str(OUTPUT / 'index.html'),
                'split': 'fixed50 test frames', 'split_sha256': manifests[0]['split_sha256'],
                'snapshot_iteration': 50000, 'comparison_images': 50,
                'panel_order': ['GT', *ARMS], 'canvas_size': [WIDTH * 4, HEIGHT + HEADER],
                'source_panel_size': [WIDTH, HEIGHT], 'source_pixels_preserved': True,
                'frames': rows}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GT + C1 + C2 + C3 | final50k</title>
<style>body{font:16px system-ui;background:#111c2c;color:#edf5ff;margin:24px}
img{width:100%;height:auto}a{color:#8de4b1}input{width:35%}button{padding:8px;margin:8px}span{margin:10px}</style>
<h1>GT | C1 repaired | C2 repaired | C3 repaired</h1>
<p>All three models: final50k. Fixed50 test frames. Each panel preserves the original1600x900 pixels;
the complete PNG is6400x950. Left to right: GT, C1, C2, C3.</p>
<button id="prev">Previous</button><input id="frame" type="range" min="0" max="49" value="0">
<button id="next">Next</button><span id="number"></span><a id="full" target="_blank">Open full-resolution PNG</a>
<a id="imageLink" target="_blank"><img id="image" alt="GT and three repaired model renders"></a>
<p><a href="manifest.json">Source and pixel-verification records</a></p>
<script>const frames=FRAME_RECORDS,f=document.getElementById('frame');
function update(){const item=frames[Number(f.value)],p=item.filename;
document.getElementById('image').src=p;document.getElementById('imageLink').href=p;
document.getElementById('full').href=p;document.getElementById('number').textContent=
(item.export_index+1)+'/50 | original frame '+String(item.original_frame_id).padStart(4,'0')+' | t='+item.normalized_time.toFixed(6);}
function step(n){f.value=Math.max(0,Math.min(49,Number(f.value)+n));update();}
f.oninput=update;document.getElementById('prev').onclick=()=>step(-1);
document.getElementById('next').onclick=()=>step(1);document.addEventListener('keydown',e=>{
if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();step(e.key==='ArrowLeft'?-1:1);}});update();
</script></html>'''
    lightweight = [{k: row[k] for k in ('export_index', 'original_frame_id', 'normalized_time', 'filename')} for row in rows]
    (OUTPUT / 'index.html').write_text(page.replace('FRAME_RECORDS', json.dumps(lightweight)), encoding='utf-8')
    (OUTPUT / 'README.md').write_text(
        '# GT + C1 + C2 + C3\n\n'
        f'Absolute directory: `{OUTPUT}`\n\n'
        '50 PNG comparisons, final50000 snapshots, fixed50 test frames.\n'
        'Left to right: GT / C1 repaired / C2 repaired / C3 repaired.\n'
        'Each panel1600x900; complete image6400x950. Source pixels remain unchanged.\n'
        '00000.png maps to original frame1; 00049.png maps to original frame197.\n'
        'Open [index.html](index.html) to browse; click an image to see its full resolution.\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'frames'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
