"""Controlled shared-time / kinematic ablation, exact resume and PNG evaluation."""
import argparse
from datetime import datetime, timedelta, timezone
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.metric_tuning import run as engine
from variants.kinematic_tuning.prepare import prepare
from variants.kinematic_tuning.timeline import build_timeline

HERE = Path(__file__).resolve().parent
RUNS = ROOT/'runs/pidg_kinematic_tuning'
REPORTS = ROOT/'experiments/kinematic_tuning'
T3_RUN = ROOT/'runs/pidg_metric_tuning/T3_baseline_perceptual/rubber_duck'
T3 = {'parent': 'W256_D0', 'pixel_loss': 'l1', 'ssim': .2, 'reg': .5, 'perceptual': .02}
PRESETS = [
    {'id': 'V1_shared_time', 'timeline': 'official_metadata', 'kinematic_weight': 0.},
    {'id': 'V2_shared_time_kinematic', 'timeline': 'official_metadata', 'kinematic_weight': .001},
    {'id': 'V3_frame_time', 'timeline': 'frame_index', 'kinematic_weight': 0.},
    {'id': 'V4_frame_time_kinematic', 'timeline': 'frame_index', 'kinematic_weight': .001},
]


def configuration(preset, iterations=50000, off=False, smoke=False):
    config, text = engine.configuration(dict(T3, id=preset['id']), iterations, off)
    config.update(timeline=preset['timeline'], shared_physics_time=not off,
                  kinematic_weight=0. if off else preset['kinematic_weight'],
                  kinematic_dt=.001, kinematic_samples=256, kinematic_interval=10,
                  kinematic_start=0 if smoke else 1000, kinematic_ramp=1 if smoke else 1000,
                  kinematic_huber_delta=.1, audit_gradients=smoke, audit_parameters=smoke)
    return config, text


def initialize(preset, iterations=50000, smoke=False, off=False):
    name = ('smoke_' if smoke else '')+preset['id']
    directory = RUNS/name/'rubber_duck'
    config, text = configuration(preset, iterations, off, smoke)
    if directory.exists():
        if engine.read_json(directory/'metric_config.json') != config or (directory/'model_config.py').read_text(encoding='utf-8') != text:
            raise ValueError('Existing experiment config changed')
        return directory
    directory.mkdir(parents=True)
    engine.write_json(directory/'metric_config.json', config)
    engine.write_text_atomic(directory/'model_config.py', text)
    if not off:
        for folder in ('point_cloud', 'deform'):
            shutil.copytree(engine.PARENTS['W256_D0']/folder/'iteration_best', directory/folder/'iteration_best')
    engine.write_json(directory/'reproduction.json', {
        'id': name, 'status': 'initialized', 'parent': config['parent_run'], 'warm_start': not off,
        'target_additional_iterations': iterations, 'optimizer_initialization': 'fresh refinement; full optimizer/RNG resume',
        'resolution_width': 400 if off else 1600, 'seed': 0, 'official_test_count': 50,
        'original_train_count': 150, 'timeline_protocol': preset['timeline'], 'tuning': config,
        'engineering_smoke_only': smoke, 'source_freeze_sha256': engine.sha256(REPORTS/'source_freeze.json'), 'commands': []})
    return directory


def history(directory):
    path = directory/'metric_history.jsonl'
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def snapshot_rows(directory, name, protocol):
    if not (directory/'metric_best.json').exists(): return []
    png = engine.read_json(directory/'results.json') if (directory/'results.json').exists() else {}
    rows = []
    for objective, best in engine.read_json(directory/'metric_best.json').items():
        key = 'ours_'+best['snapshot']
        rows.append(dict(id=name, protocol=protocol, objective=objective, iteration=best['iteration'],
                         snapshot=best['snapshot'], metrics=png.get(key, best['metrics']),
                         png_verified=key in png, run=str(directory)))
    return rows


def report():
    state = engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active = engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    live = engine.parse_progress(engine.tail(Path(active['run'])/'train.log')) if active.get('run') else {}
    rows = snapshot_rows(T3_RUN, 'V0_T3_reused', 'official_metadata')
    for p in PRESETS: rows += snapshot_rows(RUNS/p['id']/'rubber_duck', p['id'], p['timeline'])
    assessment = []
    for control, candidate in [('V1_shared_time', 'V2_shared_time_kinematic'), ('V3_frame_time', 'V4_frame_time_kinematic')]:
        a, b = RUNS/control/'rubber_duck', RUNS/candidate/'rubber_duck'
        ah, bh = history(a), history(b)
        common = sorted({r['iteration'] for r in ah} & {r['iteration'] for r in bh})
        if not common: continue
        step = common[-1]
        am = next(r['metrics'] for r in reversed(ah) if r['iteration'] == step)
        bm = next(r['metrics'] for r in reversed(bh) if r['iteration'] == step)
        final = all((d/'results.json').exists() for d in (a, b)) and step == 50000
        if final:
            am, bm = [engine.read_json(d/'results.json')['ours_50000'] for d in (a, b)]
        delta = {k: bm[k]-am[k] for k in engine.METRICS}
        improved = [k for k in engine.METRICS if engine.metric_score(delta[k], k) > (1e-4 if k == 'PSNR' else 1e-6)]
        worsened = [k for k in engine.METRICS if engine.metric_score(delta[k], k) < -(1e-4 if k == 'PSNR' else 1e-6)]
        assessment.append({'control': control, 'candidate': candidate, 'same_step': step, 'final_png': final,
                           'candidate_minus_control': delta, 'improved': improved, 'worsened': worsened,
                           'interpretation': 'pending long training' if not final else
                           'all measured metrics improved' if len(improved) == 4 else 'mixed or no consistent improvement'})
    summary = {'updated_utc': engine.now(), 'controller': state, 'active': active, 'progress': live,
               'rows': rows, 'paired_assessment': assessment,
               'evaluation': 'fixed official 50 RGB/cameras; report metadata-time and frame-index-time protocols separately',
               'selection': 'test-selected best snapshots; test RGB never enters gradient loss'}
    engine.write_json(REPORTS/'results.json', summary)
    lines = ['# Shared time and velocity / trajectory consistency', '',
             'Rubber Duck; 150 training / fixed 50 official test images; 1600x900; seed 0; additional 50k refinement.',
             'V0 reuses T3. V1/V2 share original metadata times; V3/V4 use a uniform numeric-frame timeline.',
             'Compare within each protocol. Frame-index timing assumes uniform capture cadence; physical seconds are unknown.',
             'Four best snapshots are selected using test metrics. Final 50k snapshots are also exported for equal-step comparisons.',
             'Isotropy is OFF in all arms; image objectives and model capacity match T3.', '',
             'Status: '+state.get('status', 'preparing'),
             'Current: '+active.get('id', '-')+' / '+active.get('phase', '-')+' / step '+str(live.get('iteration', '-')), '']
    for protocol in ('official_metadata', 'frame_index'):
        lines += ['## '+protocol, '', '| Model | Objective | Step | PSNR | SSIM | LPIPS | MS-SSIM | PNG verified |',
                  '|---|---|---:|---:|---:|---:|---:|---|']
        for r in rows:
            if r['protocol'] == protocol:
                lines.append('| '+' | '.join([r['id'], r['objective'], str(r['iteration']),
                    *[f"{r['metrics'][k]:.7f}" for k in engine.METRICS], str(r['png_verified'])])+' |')
        lines.append('')
    lines += ['## Paired same-step assessment', '', '```json', json.dumps(assessment, indent=2), '```', '',
              'Kinematic error is a self-consistency diagnostic, not ground-truth velocity accuracy.',
              'The original canonical-coordinate momentum residual and convective contraction are retained.',
              '[5-second monitor](monitor.html) | [Plan](plan.json) | [Preflight](preflight.json)',
              'Render and GT pairs: `'+str(RUNS/'render_comparisons')+'`', '', 'Updated UTC: '+summary['updated_utc']]
    engine.write_text_atomic(REPORTS/'results.md', '\n'.join(lines)+'\n')
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>PIDG kinematics</title><style>body{font:15px system-ui;background:#101725;color:#e5edf8;margin:30px}svg{background:#1d2b3e;margin:12px}pre{white-space:pre-wrap}a{color:#83e1c0}</style><h1>Shared time / kinematics</h1>'
    page += '<p>'+html.escape(state.get('status', 'preparing')+' | '+active.get('id', '-')+' | step '+str(live.get('iteration', '-')))+'</p><p>Refresh every 5 seconds. Metadata and frame-index protocols are separate. Loss weights / short smoke scores do not establish a benefit.</p>'
    colors = ['#f7b267', '#83e1c0', '#83bfff', '#f48fbe']
    for metric in engine.METRICS:
        curves = [(p['id'], history(RUNS/p['id']/'rubber_duck'), colors[i]) for i, p in enumerate(PRESETS)]
        values = [r['metrics'][metric] for _, h, _ in curves for r in h]
        if not values: continue
        low, high = min(values), max(values)
        span = max(high-low, 1e-5)
        page += f'<h2>{metric}</h2><svg viewBox="0 0 800 230" width="min(100%,800px)" role="img" aria-label="{metric} training evaluation">'
        page += f'<text x="10" y="20" fill="white">{high:.5f}</text><text x="10" y="210" fill="white">{low:.5f}</text>'
        for i, (name, h, color) in enumerate(curves):
            points = ' '.join(f"{70+700*r['iteration']/50000:.1f},{200-170*(r['metrics'][metric]-low)/span:.1f}" for r in h)
            page += f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/><text x="{75+i*180}" y="225" font-size="10" fill="{color}">{name}</text>'
        page += '</svg>'
    page += '<p>Full tables and interpretation: <a href="results.md">results.md</a></p><pre>'+html.escape(json.dumps(assessment, indent=2))+'</pre>'
    engine.write_text_atomic(REPORTS/'monitor.html', page)
    return summary


def export_pairs(directory):
    from PIL import Image, ImageDraw, ImageFont
    from concurrent.futures import ThreadPoolExecutor
    target = RUNS/'render_comparisons'/directory.parent.name
    target.mkdir(parents=True, exist_ok=True)
    tags = [*dict.fromkeys(engine.TAGS.values()), '50000']
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    def pair(task):
        tag, i = task
        raw = directory/'test'/('ours_'+tag)
        name = f'{i:05d}.png'
        with Image.open(raw/'gt'/name) as a, Image.open(raw/'renders'/name) as b:
            a, b = a.convert('RGB'), b.convert('RGB')
            assert a.size == b.size == (1600, 900)
            image = Image.new('RGB', (3200, 950), '#edf2f7')
            draw = ImageDraw.Draw(image)
            draw.text((15, 10), 'GT | '+name, font=font, fill='#17304a')
            draw.text((1615, 10), directory.parent.name+' | '+tag, font=font, fill='#17304a')
            image.paste(a, (0, 50)); image.paste(b, (1600, 50))
            path = target/'pairs'/tag/name
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, compress_level=3)
            with Image.open(path) as check:
                assert check.crop((0, 50, 1600, 950)).tobytes() == a.tobytes()
                assert check.crop((1600, 50, 3200, 950)).tobytes() == b.tobytes()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(pair, [(tag, i) for tag in tags for i in range(50)]))
    page = '<!doctype html><meta charset="utf-8"><title>GT / kinematic render</title><style>body{background:#101725;color:white;font:16px system-ui;margin:25px}img{width:100%}</style><h1>'+directory.parent.name+'</h1><p>Left: GT; right: render. Full resolution, 50 official test frames.</p><select id="tag">'
    page += ''.join('<option>'+tag+'</option>' for tag in tags)+'</select><input id="frame" type="range" min="0" max="49" value="25"><span id="number"></span><a id="full">Open pair</a><img id="image"><script>const t=document.getElementById("tag"),f=document.getElementById("frame");function update(){let n=String(f.value).padStart(5,"0"),p="pairs/"+t.value+"/"+n+".png";document.getElementById("image").src=p;document.getElementById("full").href=p;document.getElementById("number").textContent=n}t.onchange=f.oninput=update;update()</script>'
    engine.write_text_atomic(target/'index.html', page)
    engine.write_json(target/'manifest.json', {'run_absolute': str(directory), 'gallery_absolute': str(target/'index.html'),
                      'pairs_absolute': str(target/'pairs'), 'pair_count': 250, 'pixel_identity_verified': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--hours', type=float, default=24)
    args = parser.parse_args()
    if args.hours <= 0: raise ValueError('Positive time budget required')
    RUNS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    engine.assert_frozen(engine.prepare())
    repo = prepare()
    original_freeze = engine.freeze
    def freeze(path):
        value = original_freeze(path)
        value['kinematic_helpers'] = {str(p.relative_to(ROOT)): engine.sha256(p) for p in HERE.glob('*.py')}
        value['historical_T3'] = {str(p): engine.sha256(p) for p in [T3_RUN/'metric_best.json', T3_RUN/'results.json', T3_RUN/'metric_history.jsonl']}
        return value
    engine.RUNS, engine.REPORTS, engine.repo_global, engine.freeze = RUNS, REPORTS, repo, freeze
    frozen = freeze(repo)
    if (REPORTS/'source_freeze.json').exists():
        if engine.read_json(REPORTS/'source_freeze.json') != frozen: raise ValueError('Experiment source changed')
    else: engine.write_json(REPORTS/'source_freeze.json', frozen)
    engine.write_json(REPORTS/'timeline_audit.json', build_timeline(ROOT/'data/PIDG/rubber_duck'))
    for p in PRESETS:
        config, text = configuration(p)
        engine.write_json(REPORTS/'configs'/(p['id']+'.json'), config)
        engine.write_text_atomic(REPORTS/'configs'/(p['id']+'.py'), text)
    engine.write_json(REPORTS/'plan.json', {'presets': PRESETS, 'hours': args.hours, 'additional_steps_per_arm': 50000,
        'parent': str(engine.PARENTS['W256_D0']), 'control_reused': str(T3_RUN), 'isotropic_weight': 0,
        'loss': 'T3 + 0.001 * ramp * interval * Huber(v(X,tau), d[R(q(X,tau))X+b(X,tau)]/dtau; delta=0.1)',
        'derivative': 'second-order finite differences h=0.001; one-sided endpoints; joint parameter gradients',
        'time': 'normalized timeline [0,1]; velocity in scene units per normalized time; physical seconds unknown',
        'evaluation': 'fixed official 50 images/poses, two explicitly separate time-input protocols; PNG verification',
        'limitations': ['same canonical-coordinate legacy momentum residual', 'single scene and seed', 'test-selected snapshots',
                        'frame-index clock assumes uniform cadence; retimed weights warm-start from metadata-time parent']})
    deadline = datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state = {'pid': os.getpid(), 'started_utc': engine.now(), 'deadline_utc': deadline.isoformat(),
             'status': 'validating' if args.smoke else 'training', 'mode': 'smoke' if args.smoke else 'long', 'completed': []}
    stop = threading.Event()
    def monitor():
        while not stop.wait(5):
            try: report()
            except (OSError, ValueError): pass
    with engine.lock():
        if (RUNS/'stop_queue.request').exists(): raise RuntimeError('Pause request remains')
        engine.write_json(RUNS/'controller.json', state); report()
        thread = threading.Thread(target=monitor, daemon=True); thread.start()
        try:
            if args.smoke:
                off = initialize(dict(PRESETS[0], id='flags_off'), 20, True, True)
                assert engine.train(off, repo, deadline, 20, off=True, phase='flags_off') == 0
                control = initialize(PRESETS[2], 100, True)
                assert engine.train(control, repo, deadline, 100, phase='smoke') == 0
                joint = initialize(PRESETS[3], 100, True)
                step = engine.read_json(joint/'checkpoints/resume.json')['iteration'] if (joint/'checkpoints/resume.json').exists() else 0
                if step < 20:
                    assert engine.train(joint, repo, deadline, 100, limit=20, phase='smoke') == 75
                    assert engine.read_json(joint/'checkpoints/resume.json')['iteration'] == 20
                if step < 100: assert engine.train(joint, repo, deadline, 100, phase='resume_check') == 0
                assert engine.read_json(joint/'checkpoints/resume.json')['iteration'] == 100
                gradients = [json.loads(s) for s in (joint/'kinematic_parameter_gradient.jsonl').read_text().splitlines()]
                assert {'grid', 'deformation', 'velocity'} == {g['branch'] for g in gradients}
                assert all(g['finite'] and g['gradient_l1'] > 0 for g in gradients)
                engine.render_run(joint, repo, deadline, tags=['best'])
                engine.write_json(REPORTS/'preflight.json', {'status': 'passed', 'flags_off_steps': 20,
                    'shared_time_control_steps': 100, 'joint_steps': 100, 'full_resume_from': 20,
                    'finite_nonzero_actual_parameter_gradients': gradients, 'momentum_step_100_completed': True,
                    'retimed_train_eval_render_png_parity': True, 'completed_utc': engine.now(),
                    'quality_improvement_not_claimed': True})
                state['status'] = 'preflight_passed'
            else:
                if engine.read_json(REPORTS/'preflight.json')['status'] != 'passed': raise ValueError('Preflight required')
                for p in PRESETS:
                    if datetime.now(timezone.utc) >= deadline-timedelta(minutes=30):
                        state['status'] = 'budget_stopped'; break
                    directory = initialize(p)
                    step = engine.read_json(directory/'checkpoints/resume.json')['iteration'] if (directory/'checkpoints/resume.json').exists() else 0
                    if step < 50000:
                        if engine.train(directory, repo, deadline-timedelta(minutes=30), phase='long') == 75:
                            state['status'] = 'saved_stopped'; break
                    engine.render_run(directory, repo, deadline, tags=[*dict.fromkeys(engine.TAGS.values()), '50000'])
                    export_pairs(directory)
                    state['completed'].append(p['id'])
                    engine.write_json(RUNS/'controller.json', state)
                else: state['status'] = 'completed'
            engine.assert_frozen(repo)
        except SystemExit as error:
            state['status'] = 'saved_stopped' if error.code == 75 else 'failed'
            if error.code != 75: raise
        except Exception as error:
            state.update(status='failed', error=str(error)); raise
        finally:
            state['finished_utc'] = engine.now()
            engine.write_json(RUNS/'controller.json', state)
            stop.set(); thread.join(timeout=6); report()


if __name__ == '__main__': main()
