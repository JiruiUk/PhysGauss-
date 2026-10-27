"""Watch the existing repair queue and collect final-test renders and GT pairs.

CPU-only: consumes completed PNG exports; never starts training, renders on GPU,
reads source dataset RGB, or changes model-selection files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import time

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs/pidg_color_repair'
REPORTS = ROOT / 'experiments/color_repair'
OUTPUT = RUNS / 'render_exports/final50000'
ARMS = ('C1_repaired', 'C2_repaired', 'C3_repaired')
SPLIT = ROOT / 'data/derived/kinematic_resplit/final_test/transforms_test.json'
EXPECTED = {f'{i:05d}.png' for i in range(50)}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def write(path, value):
    atomic_text(path, json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum = digest(source)
    if target.exists():
        if digest(target) != checksum:
            raise ValueError(f'Existing output differs from source: {target}')
    else:
        temporary = target.with_name(target.name + '.tmp')
        shutil.copyfile(source, temporary)
        if digest(temporary) != checksum:
            raise ValueError('Image copy changed source bytes')
        temporary.replace(target)
    return checksum


def export_frame(raw, output, arm, index, frame):
    filename = f'{index:05d}.png'
    original_id = int(Path(frame['file_path']).name)
    gt_path = output / 'gt' / filename
    render_path = output / arm / 'renders' / filename
    gt_hash = copy_exact(raw / 'gt' / filename, gt_path)
    render_hash = copy_exact(raw / 'renders' / filename, render_path)
    with Image.open(gt_path) as a, Image.open(render_path) as b:
        a, b = a.convert('RGB'), b.convert('RGB')
        if a.size != b.size or a.size != (1600, 900):
            raise ValueError('Expected unscaled1600x900 source images')
        image = Image.new('RGB', (3200, 950), '#edf2f7')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 25)
        draw.text((15, 10), f'GT | test frame {original_id:04d}', font=font, fill='#17304a')
        draw.text((1615, 10), f'{arm} | final50000', font=font, fill='#17304a')
        image.paste(a, (0, 50)); image.paste(b, (1600, 50))
        pair = output / arm / 'gt_render' / filename
        pair.parent.mkdir(parents=True, exist_ok=True)
        temporary = pair.with_name(pair.stem + '.tmp.png')
        image.save(temporary, compress_level=3)
        with Image.open(temporary) as check:
            if check.crop((0, 50, 1600, 950)).tobytes() != a.tobytes() or \
               check.crop((1600, 50, 3200, 950)).tobytes() != b.tobytes():
                raise ValueError('Concatenation changed source pixels')
        temporary.replace(pair)
    return {'export_index': index, 'original_frame_id': original_id,
            'normalized_time': (original_id - 1) / 199,
            'filename': filename, 'gt_sha256': gt_hash, 'render_sha256': render_hash,
            'pair_sha256': digest(pair)}


def ready(arm):
    lock = read(RUNS / 'selection_lock.json', {})
    if not set(ARMS).issubset(lock.get('validation_selections', {})):
        return False
    directory = RUNS / arm / 'rubber_duck'
    checkpoint = read(directory / 'checkpoints/resume.json', {})
    raw = directory / 'test/ours_50000'
    receipt = read(raw / 'render_complete.json', {})
    return checkpoint.get('iteration') == 50000 and receipt.get('split') == 'test' and \
        receipt.get('images') == 50 and all(
            {p.name for p in (raw / kind).glob('*.png')} == EXPECTED
            for kind in ('gt', 'renders'))


def export_arm(arm, frames):
    source = RUNS / arm / 'rubber_duck'
    raw = source / 'test/ours_50000'
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda item: export_frame(raw, OUTPUT, arm, *item), enumerate(frames)))
    manifest = {'status': 'completed', 'arm': arm, 'iteration': 50000, 'split': 'test',
                'images': 50, 'created_utc': now(), 'source_run': str(source),
                'source_render_receipt_sha256': digest(raw / 'render_complete.json'),
                'source_checkpoint_sha256': read(source / 'checkpoints/resume.json')['sha256'],
                'split_sha256': digest(SPLIT), 'render_size': [1600, 900], 'pair_size': [3200, 950],
                'render_directory': str(OUTPUT / arm / 'renders'),
                'pair_directory': str(OUTPUT / arm / 'gt_render'),
                'source_image_bytes_preserved': True, 'pair_pixels_verified': True, 'frames': records}
    write(OUTPUT / arm / 'manifest.json', manifest)
    return manifest


def publish(state, frames):
    state['updated_utc'] = now()
    write(REPORTS / 'render_export_status.json', state)
    write(OUTPUT / 'status.json', state)
    complete = [arm for arm in ARMS if arm in state['completed']]
    text = ['# C1 / C2 / C3 repaired final50k image exports', '',
            f'Status: {state["status"]}', '', f'Absolute output directory: `{OUTPUT}`', '',
            'Fixed50 test frames, original IDs1,5,...197. Final50000 snapshots.',
            'Per model:50 independent render PNGs and50 GT/render pairs.',
            'Render1600x900; pair3200x950, left GT and right render. No rescaling or recoloring.',
            'The shared `gt` folder contains50 identical ground-truth images.', '',
            '| Model | Status | Render PNG | GT/render PNG |', '|---|---|---:|---:|']
    for arm in ARMS:
        done = arm in complete
        text.append(f'| {arm} | {"completed" if done else "waiting"} | {50 if done else 0} | {50 if done else 0} |')
    if state.get('error'):
        text += ['', 'Error: ' + state['error']]
    text += ['', '[Open gallery](../../runs/pidg_color_repair/render_exports/final50000/index.html)',
             '', 'Updated UTC: ' + state['updated_utc']]
    atomic_text(REPORTS / 'render_exports.md', '\n'.join(text) + '\n')
    atomic_text(OUTPUT / 'README.md', ('\n'.join(text) + '\n').replace(
        '../../runs/pidg_color_repair/render_exports/final50000/index.html', 'index.html'))
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    if state['status'] != 'completed':
        page += '<meta http-equiv="refresh" content="5">'
    page += '<title>C1 / C2 / C3 final50k renders</title><style>body{font:16px system-ui;background:#111c2c;color:#edf5ff;margin:25px}img{width:100%;height:auto}a{color:#8de4b1}select,button{padding:7px;margin:5px}input{width:35%}</style>'
    page += '<h1>C1 / C2 / C3 repaired: final50k</h1><p>Fixed50 test frames. GT/render pairs: left GT, right render.</p><p>'
    page += html.escape(state['status']) + f' | {len(complete)}/3 models exported.</p>'
    for arm in ARMS:
        page += '<p>' + arm + (f': <a href="{arm}/renders/">renders</a> | <a href="{arm}/gt_render/">GT/render pairs</a>' if arm in complete else ': waiting for completed test renders') + '</p>'
    if complete:
        page += '<select id="arm">' + ''.join(f'<option>{a}</option>' for a in complete) + '</select>'
        page += '<select id="kind"><option value="gt_render">GT / render</option><option value="renders">Render only</option><option value="gt">GT only</option></select><button id="prev">Previous</button><input id="frame" type="range" min="0" max="49" value="0"><button id="next">Next</button><span id="number"></span> <a id="full">Open PNG</a><img id="image">'
        page += '<script>const ids=' + json.dumps([int(Path(f['file_path']).name) for f in frames]) + ';'
        page += '''const a=document.getElementById('arm'),k=document.getElementById('kind'),f=document.getElementById('frame');
function update(){const i=Number(f.value),n=String(i).padStart(5,'0'),p=(k.value==='gt'?'gt':a.value+'/'+k.value)+'/'+n+'.png';
document.getElementById('image').src=p;document.getElementById('full').href=p;
document.getElementById('number').textContent=' Original frame '+ids[i]+' | '+(i+1)+'/50';}
function step(n){f.value=Math.max(0,Math.min(49,Number(f.value)+n));update();}
a.onchange=k.onchange=f.oninput=update;document.getElementById('prev').onclick=()=>step(-1);
document.getElementById('next').onclick=()=>step(1);update();</script>'''
    page += '<p>Updated UTC: ' + state['updated_utc'] + '</p></html>'
    atomic_text(OUTPUT / 'index.html', page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--wait-hours', type=float, default=26.)
    args = parser.parse_args()
    if args.wait_hours <= 0:
        raise ValueError('Positive wait limit required')
    frames = read(SPLIT)['frames']
    if [int(Path(f['file_path']).name) for f in frames] != list(range(1, 201, 4)):
        raise ValueError('Unexpected official test frame order')
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = REPORTS / 'render_export.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    state = {'pid': os.getpid(), 'started_utc': now(), 'status': 'waiting',
             'output_absolute': str(OUTPUT), 'gallery_absolute': str(OUTPUT / 'index.html'),
             'completed': {}, 'expected_render_images': 150, 'expected_pair_images': 150,
             'poll_seconds': 5, 'snapshot_iteration': 50000, 'split': 'test'}
    deadline = datetime.now(timezone.utc) + timedelta(hours=args.wait_hours)
    try:
        while True:
            controller = read(RUNS / 'controller.json', {})
            state['queue_status'] = controller.get('status', 'unknown')
            state['status'] = 'waiting_for_training' if state['queue_status'] == 'training' else 'waiting_for_test_renders'
            for arm in ARMS:
                if arm not in state['completed'] and ready(arm):
                    state.update(status='exporting', active=arm); publish(state, frames)
                    print('Exporting ' + arm, flush=True)
                    state['completed'][arm] = export_arm(arm, frames)
            state.pop('active', None)
            if len(state['completed']) == 3:
                state['status'] = 'completed'
                state['finished_utc'] = now()
                publish(state, frames)
                print('Completed: ' + str(OUTPUT), flush=True)
                return
            publish(state, frames)
            if state['queue_status'] == 'failed':
                raise RuntimeError('Training/render queue failed: ' + str(controller.get('error', 'see queue logs')))
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError('Timed out waiting for completed official test renders')
            if not args.watch:
                print('Not all test renders are ready; use --watch to export automatically.', flush=True)
                return
            time.sleep(5)
    except Exception as error:
        state.update(status='failed', error=str(error)); publish(state, frames)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
