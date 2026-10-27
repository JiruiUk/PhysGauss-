"""Export a completed repair's validation preview without opening final test."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from variants.color_repair import run as repair
from variants.metric_tuning import run as engine


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=repair.NAMES, default='C1_repaired')
    args = parser.parse_args()
    repair.assert_frozen(repair.HERE / 'code')
    source = repair.RUNS / args.arm / 'rubber_duck'
    if engine.read_json(source / 'checkpoints/resume.json')['iteration'] != 50000:
        raise ValueError('Only a completed final50k model may be previewed')
    target = repair.RUNS / 'validation_previews' / (args.arm + '_50000')
    files = [source / name for name in ('cfg_args', 'model_config.py', 'metric_config.json',
                                       'metric_best.json', 'metric_history.jsonl')]
    folders = ('point_cloud/iteration_50000', 'deform/iteration_50000')
    files += [p for folder in folders for p in (source / folder).rglob('*') if p.is_file()]
    hashes = {str(p.relative_to(source)): digest(p) for p in files}
    receipt = target / 'input_manifest.json'
    if receipt.exists() and engine.read_json(receipt)['source_sha256'] != hashes:
        raise ValueError('Existing preview belongs to different source weights or selections')
    target.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        shutil.copytree(source / folder, target / folder, dirs_exist_ok=True)
    for name in ('cfg_args', 'model_config.py', 'metric_config.json'):
        shutil.copyfile(source / name, target / name)
    engine.write_json(receipt, {'source': str(source), 'source_sha256': hashes,
                                'split': 'validation', 'final_test_opened': False})
    repo = repair.HERE / 'code'
    environment = engine.env_for(target)
    environment['PIDG_EXPORT_ALPHA'] = '1'
    prefix = [sys.executable, '-u', str(ROOT / 'scripts/pidg_variant_entrypoint.py'),
              '--repo', str(repo), '--seed', '0']
    commands = [
        ('render', [*prefix, 'render.py', '-s', str(repair.base.DATA / 'training'),
                    '-m', str(target), '--conf', str(target / 'model_config.py'),
                    '--load2gpu_on_the_fly', '--resolution', '1600',
                    '--iteration', '50000', '--skip_train']),
        ('metrics', [*prefix, 'metrics.py', '-m', str(target)]),
    ]
    for phase, command in commands:
        print(f'{phase}: {args.arm}, final50000, validation30', flush=True)
        with (target / (phase + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=repo, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError(f'{phase} failed; see {target / (phase + ".log")}')
    raw = target / 'test/ours_50000'  # PIDG calls the held-out slot "test" even for val.
    expected = {f'{i:05d}.png' for i in range(30)}
    for folder in ('renders', 'gt', 'alpha'):
        if {p.name for p in (raw / folder).glob('*.png')} != expected:
            raise ValueError(f'Incomplete validation export: {folder}')
    results = engine.read_json(target / 'results.json')
    per_view = engine.read_json(target / 'per_view.json')
    repair.base.verify_metrics(results, per_view, ['50000'], 30)
    reference = next(json.loads(line)['metrics'] for line in
                     (source / 'metric_history.jsonl').read_text().splitlines()
                     if json.loads(line)['iteration'] == 50000)
    metrics = results['ours_50000']
    if any(abs(metrics[k] - reference[k]) > 1e-4 for k in engine.METRICS):
        raise ValueError('Exported PNG metrics disagree with training validation')
    frames = engine.read_json(repair.base.DATA / 'training/transforms_test.json')['frames']
    if len(frames) != 30:
        raise ValueError('Expected30 validation frames')
    pairs = target / 'pairs'
    pairs.mkdir(exist_ok=True)

    def pair(index):
        filename = f'{index:05d}.png'
        original_id = int(Path(frames[index]['file_path']).name)
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 25)
        with Image.open(raw / 'gt' / filename) as a, Image.open(raw / 'renders' / filename) as b:
            a, b = a.convert('RGB'), b.convert('RGB')
            if a.size != b.size or a.size != (1600, 900):
                raise ValueError('Unexpected full-resolution dimensions')
            image = Image.new('RGB', (3200, 950), '#edf2f7')
            draw = ImageDraw.Draw(image)
            draw.text((15, 10), f'GT | validation frame {original_id:04d}', font=font, fill='#17304a')
            draw.text((1615, 10), f'{args.arm} | final50000', font=font, fill='#17304a')
            image.paste(a, (0, 50)); image.paste(b, (1600, 50))
            output = pairs / filename
            image.save(output, compress_level=3)
            with Image.open(output) as check:
                if check.crop((0, 50, 1600, 950)).tobytes() != a.tobytes() or \
                   check.crop((1600, 50, 3200, 950)).tobytes() != b.tobytes():
                    raise ValueError('Pair changed source pixels')
        return {'export_index': index, 'original_frame_id': original_id,
                'pair': str(output), 'sha256': digest(output)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(pair, range(30)))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>ARM: validation GT / render</title>
<style>body{font:16px system-ui;background:#111c2c;color:#edf5ff;margin:24px}
img{width:100%;height:auto}a{color:#8de4b1}input{width:40%}span{margin:15px}</style>
<h1>ARM: final 50k validation preview</h1>
<p>30 validation frames. Left: GT; right: render. Each source is1600 x900.
These are validation images, not the50 final-test images. Final test remains locked.</p>
<p>METRICS</p><input id="frame" type="range" min="0" max="29" value="0">
<span id="number"></span><a id="full">Open full-resolution pair</a><img id="image">
<script>const ids=IDS,f=document.getElementById('frame');
function update(){const i=Number(f.value),n=String(i).padStart(5,'0'),p='pairs/'+n+'.png';
document.getElementById('image').src=p;document.getElementById('full').href=p;
document.getElementById('number').textContent='Original frame '+String(ids[i]).padStart(4,'0')+' | '+(i+1)+'/30';}
f.oninput=update;update();</script></html>'''
    page = page.replace('ARM', html.escape(args.arm)).replace('IDS', json.dumps([r['original_frame_id'] for r in rows]))
    page = page.replace('METRICS', 'Validation: ' + ' | '.join(f'{k} {metrics[k]:.6f}' for k in engine.METRICS))
    engine.write_text_atomic(target / 'index.html', page)
    if any(digest(source / name) != checksum for name, checksum in hashes.items()):
        raise ValueError('Original model weights or selections changed')
    repair.assert_frozen(repo)
    manifest = {'status': 'completed', 'split': 'validation', 'images': 30, 'iteration': 50000,
                'arm': args.arm, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'gallery_absolute': str(target / 'index.html'), 'pairs_absolute': str(pairs),
                'raw_renders_absolute': str(raw / 'renders'), 'metrics': metrics,
                'validation_png_parity': True, 'pair_pixels_verified': True,
                'final_test_opened': False, 'source_unchanged': True, 'frames': rows}
    engine.write_json(target / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'frames'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
