"""Two genuinely from-scratch 50k runs with one identical random step-zero state."""
import argparse
from datetime import datetime, timedelta, timezone
import html
import json
import os
from pathlib import Path
import shutil
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.v3_iso_scratch.prepare import prepare

HERE = Path(__file__).resolve().parent
RUNS = ROOT/'runs/pidg_v3_iso_scratch'
REPORTS = ROOT/'experiments/v3_iso_scratch'
NAMES = ['V3_scratch_control', 'V3_scratch_iso_1e3']
TAGS = base.TAGS


def configuration(name, smoke=False, off=False):
    config = {'enabled': not off, 'parent_run': '', 'pixel_loss': 'l1', 'structure': 'ssim',
        'perceptual_weight': .02, 'perceptual_interval': 4, 'timeline': 'frame_index',
        'shared_physics_time': True, 'kinematic_weight': 0.,
        'isotropic_weight': .001 if name == NAMES[1] and not off else 0.,
        'isotropic_scope': 'rendered', 'isotropic_eps': 1e-8, 'isotropic_start': 0,
        'engineering_smoke': smoke, 'allow_initial_capture': True}
    text = (engine.SOURCE/'arguments/pidg/rubber_dack.py').read_text(encoding='utf-8')
    text += '\nnetwork_args = dict(network_args, width=256, depth=0, metric_residual_width=0)\n'
    # Full initial optimization, rather than the small-LR continuation recipe.
    text += '\nwarm_up = 3000\ndensify_until_iter = 15000\n'
    if smoke: text += '\nwarm_up = 0\ndensify_until_iter = 0\nposition_lr_max_steps = 100\ndeform_lr_max_steps = 100\n'
    return config, text


def branch_config(a, b):
    a, b = dict(a), dict(b)
    if a.pop('isotropic_weight') != 0. or b.pop('isotropic_weight') not in (0., .001) or a != b:
        raise ValueError('Only the predeclared iso weight may change from the random control state')


def validate_initial_state(state):
    if state['iteration'] != 0 or state['gaussian_optimizer']['state'] or state['deform_optimizer']['state']:
        raise ValueError('Initial state must be step zero with no optimizer updates')


def initialize(name, smoke=False, off=False):
    directory = RUNS/('_smoke' if smoke else '')/name/'rubber_duck'
    config, text = configuration(name, smoke, off)
    if (directory/'reproduction.json').exists():
        if engine.read_json(directory/'metric_config.json') != config or (directory/'model_config.py').read_text() != text:
            raise ValueError('Existing scratch config changed')
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    engine.write_json(directory/'metric_config.json', config)
    engine.write_text_atomic(directory/'model_config.py', text)
    engine.write_json(directory/'reproduction.json', {'id': name, 'status': 'initialized', 'warm_start': False,
        'parent': None, 'source': 'fresh random network, Gaussian and velocity/stress parameters; shared random geometry',
        'target_iterations': 100 if smoke else 50000, 'train': 120, 'validation': 30, 'test': 50,
        'seed': 0, 'resolution_width': 400 if smoke else 1600,
        'source_freeze_sha256': engine.sha256(REPORTS/'source_freeze.json'), 'commands': []})
    return directory


def freeze(repo):
    inputs = [*HERE.glob('*.py'), ROOT/'variants/isotropic_tuning/model.py', ROOT/'variants/isotropic_tuning/runtime.py',
              ROOT/'variants/v3_isotropic/runtime.py', ROOT/'variants/v3_isotropic/prepare.py']
    return {'resplit_source_data': base.freeze(base.HERE/'code'), 'code': engine.runtime_hashes(repo),
            'inputs': {str(p.relative_to(ROOT)): engine.sha256(p) for p in inputs}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json') != freeze(repo):
        raise ValueError('Scratch source or data changed')


def wire(repo):
    base.RUNS, base.REPORTS, base.assert_frozen = RUNS, REPORTS, assert_frozen
    engine.RUNS, engine.REPORTS, engine.repo_global, engine.assert_frozen = RUNS, REPORTS, repo, assert_frozen


def clone_zero(source, destination):
    if (destination/'branch.json').exists(): return
    checkpoint = destination/'checkpoints/resume.pt'
    if checkpoint.exists(): raise ValueError('Unreceipted initial state exists')
    a, b = [engine.read_json(d/'metric_config.json') for d in (source, destination)]
    branch_config(a, b)
    from reproduction.scripts.training_checkpoint import load_state
    validate_initial_state(load_state(source/'checkpoints/resume.pt'))
    manifest = engine.read_json(source/'checkpoints/resume.json')
    if manifest['iteration'] != 0: raise ValueError('Source has already trained')
    for folder in ('point_cloud', 'deform', 'checkpoints'):
        shutil.copytree(source/folder, destination/folder, dirs_exist_ok=True)
    for name in ('metric_best.json', 'metric_history.jsonl'):
        shutil.copyfile(source/name, destination/name)
    state = engine.read_json(checkpoint.with_suffix('.metrics.json'))
    if state['config'] != a: raise ValueError('Initial config sidecar changed')
    state['config'] = b
    engine.write_json(checkpoint.with_suffix('.metrics.json'), state)
    digest = engine.sha256(checkpoint)
    if digest != manifest['sha256']: raise ValueError('Step-zero tensors were not copied identically')
    engine.write_json(destination/'branch.json', {'source': str(source), 'destination': str(destination), 'iteration': 0,
        'identical_full_checkpoint_sha256': digest, 'source_config': a, 'branch_config': b,
        'optimizer_updates_before_branch': 0, 'pretrained_model_used': False,
        'only_change': 'isotropic_weight: '+str(a['isotropic_weight'])+' -> '+str(b['isotropic_weight']),
        'created_utc': engine.now()})


def report():
    state = engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active = engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    rows = []
    for name in NAMES:
        p = RUNS/name/'rubber_duck'
        rows.append({'id': name, 'iteration': base.progress(p), 'history': base.history(p),
                     'test': engine.read_json(p/'results.json') if (p/'results.json').exists() else None})
    paired = None
    if all(r['test'] for r in rows):
        a, b = [r['test']['ours_50000'] for r in rows]
        views = [engine.read_json(RUNS/n/'rubber_duck/per_view.json')['ours_50000'] for n in NAMES]
        paired = {'iso_minus_control': {k: b[k]-a[k] for k in engine.METRICS},
            'improved_frames': {k: sum(engine.metric_score(views[1][k][n]-views[0][k][n], k)>0 for n in views[0][k])
                                for k in engine.METRICS}, 'test_frames': 50}
    result = {'updated_utc': engine.now(), 'controller': state, 'active': active, 'rows': rows,
        'protocol': 'FROM RANDOM INITIALIZATION; 50k total per arm; 120 train/30 validation/50 test; iso=0 vs .001 from step1',
        'paired_final50k': paired, 'pretrained_model_used': False,
        'limitations': ['single scene/seed', 'historical test exposure persists', 'uniform frame cadence assumed',
                        'fresh50k differs from prior two-stage V3; compare the two scratch arms to isolate iso',
                        'identical initial state does not eliminate CUDA nondeterminism after training starts']}
    engine.write_json(REPORTS/'results.json', result)
    lines = ['# V3 / V3+iso: from-scratch 50k', '', result['protocol'], '',
        'Status: '+state.get('status', 'preparing'), 'Active: '+active.get('id', '-')+' / '+active.get('phase', '-'), '',
        'No trained model is loaded. Both arms copy the same full step-zero state with empty optimizer histories.',
        'V3 image losses/shared time; original PIDG initial-training optimization, warm-up3000 and densification through15000.',
        'Iso weight0.001 is computed from step1. Validation alone selects checkpoints; test stays locked until both arms finish.', '',
        '| Model | Total training iter | Latest validation PSNR |', '|---|---:|---:|']
    for r in rows:
        value = r['history'][-1]['metrics']['PSNR'] if r['history'] else '-'
        lines.append(f"| {r['id']} | {r['iteration']}/50000 | {value} |")
    lines += ['', '## Fixed final50k test results', '', '| Model | PSNR | SSIM | LPIPS | MS-SSIM |', '|---|---:|---:|---:|---:|']
    for r in rows:
        if r['test']:
            m = r['test']['ours_50000']
            lines.append('| '+r['id']+' | '+' | '.join(f'{m[k]:.7f}' for k in engine.METRICS)+' |')
    if paired:
        lines += ['', '| Metric | Iso minus control | Improved frames / 50 |', '|---|---:|---:|']
        for k in engine.METRICS:
            lines.append(f"| {k} | {paired['iso_minus_control'][k]:+.7f} | {paired['improved_frames'][k]} |")
    lines += ['', '## Validation-selected snapshots: final test scores', '',
              '| Model | Validation objective | Selected iter | PSNR | SSIM | LPIPS | MS-SSIM |', '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if not r['test']: continue
        for objective, best in engine.read_json(RUNS/r['id']/'rubber_duck/metric_best.json').items():
            m = r['test']['ours_'+best['snapshot']]
            lines.append('| '+' | '.join([r['id'], objective, str(best['iteration']),
                *[f'{m[k]:.7f}' for k in engine.METRICS]])+' |')
    lines += ['', 'This replaces the paused continuation experiment; no continuation checkpoint is used.',
        'The same official test set has historical exposure; prospective separation does not undo that history.',
        '[5-second graphical monitor](monitor.html) | [Plan](plan.json) | [Initial state](initialization.json) | [GPU preflight](preflight.json)',
        '', 'GT/render pairs absolute directory: '+str(RUNS/'render_comparisons'), '', 'Updated UTC: '+result['updated_utc']]
    engine.write_text_atomic(REPORTS/'results.md', '\n'.join(lines)+'\n')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>V3 iso FROM SCRATCH</title><style>body{font:16px system-ui;background:#111c2c;color:#e5edf8;margin:30px}section,svg{background:#203047;padding:15px;margin:10px 0}progress{width:80%}a{color:#82e6bd}svg{max-width:100%}</style><h1>V3 / V3+iso: FROM SCRATCH</h1><p>50k TOTAL per model. Same random step-zero state; no pretrained weights. Refresh: 5 seconds.</p><p>120 train / 30 validation / 50 test. Curves show validation only. Iso starts at step1.</p><p>'+html.escape(state.get('status', 'preparing')+' | '+active.get('id', '-')+' | '+active.get('phase', '-'))+'</p>'
    for r in rows:
        page += f"<section><b>{r['id']}</b><h2>{r['iteration']:,} / 50,000 total iter</h2><progress value='{r['iteration']}' max='50000'></progress></section>"
    for key in engine.METRICS:
        values = [p['metrics'][key] for r in rows for p in r['history']]
        if not values: continue
        lo, hi = min(values), max(values)
        span = max(hi-lo, 1e-6)
        page += f'<h2>Validation {key}</h2><svg viewBox="0 0 900 260" width="900"><text x="8" y="22" fill="white">{hi:.5f}</text><text x="8" y="220" fill="white">{lo:.5f}</text>'
        for i, (r, color) in enumerate(zip(rows, ['#8fd3ff', '#82e6bd'])):
            points = ' '.join(f"{80+790*p['iteration']/50000:.1f},{220-190*(p['metrics'][key]-lo)/span:.1f}" for p in r['history'])
            page += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{80+i*400}" y="250" fill="{color}" font-size="13">{r["id"]}</text>'
        page += '</svg>'
    page += '<p><a href="results.md">Full results and protocol</a></p></html>'
    engine.write_text_atomic(REPORTS/'monitor.html', page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--hours', type=float, default=24)
    args = parser.parse_args()
    if args.hours <= 0: raise ValueError('Positive budget required')
    RUNS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    parent_repo = base.prepare()
    if engine.read_json(ROOT/'experiments/kinematic_resplit/source_freeze.json') != base.freeze(parent_repo):
        raise ValueError('Frozen dataset or parent source changed')
    repo = prepare()
    if (REPORTS/'source_freeze.json').exists(): assert_frozen(repo)
    else: engine.write_json(REPORTS/'source_freeze.json', freeze(repo))
    for name in NAMES:
        config, text = configuration(name)
        engine.write_json(REPORTS/'configs'/(name+'.json'), config)
        engine.write_text_atomic(REPORTS/'configs'/(name+'.py'), text)
    engine.write_json(REPORTS/'plan.json', {'initialization': 'identical random full state at step0; no trained parent',
        'parent_run': None, 'total_steps_per_arm': 50000, 'arms': dict(zip(NAMES, [0., .001])),
        'iso_first_computed_step': 1, 'kinematic_weight': 0., 'train_validation_test': [120, 30, 50],
        'optimizer': 'original PIDG initial-training schedule, not continuation; warm-up3000/densification15000',
        'image_loss': 'V3: 0.8 L1 + 0.2(1-SSIM) + 0.02 LPIPS (every4 with interval compensation)',
        'time': '(original_frame_id-1)/199; same deformation/physics input',
        'selection': 'validation only; lock both arms before test', 'test_tags': TAGS,
        'seed': 0, 'resolution': [1600, 900], 'budget_hours': args.hours,
        'limitations': ['prior official-test exposure', 'uniform cadence assumed', 'single seed', 'CUDA nondeterminism']})
    wire(repo)
    os.environ.pop('PIDG_CAPTURE_INITIAL_ONLY', None)
    deadline = datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state = {'pid': os.getpid(), 'status': 'preflight' if args.smoke else 'initializing_from_random',
             'started_utc': engine.now(), 'deadline_utc': deadline.isoformat(), 'completed': []}
    stop = threading.Event()
    def monitor():
        while not stop.wait(5):
            try: report()
            except (OSError, ValueError, KeyError): pass
    def step(d):
        return engine.read_json(d/'checkpoints/resume.json')['iteration'] if (d/'checkpoints/resume.json').exists() else -1
    with engine.lock():
        if (RUNS/'stop_queue.request').exists(): raise RuntimeError('Pause request remains')
        engine.write_json(RUNS/'controller.json', state); report()
        watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
        try:
            if not args.smoke and engine.read_json(REPORTS/'preflight.json')['status'] != 'passed': raise ValueError('GPU preflight required')
            if args.smoke:
                off = initialize('flags_off', True, True)
                if step(off)<20: assert base.train(off, repo, deadline, smoke=True, off=True)==0
            initial = initialize('initialization', args.smoke)
            if step(initial)<0:
                os.environ['PIDG_CAPTURE_INITIAL_ONLY'] = '1'
                try: assert base.train(initial, repo, deadline, args.smoke)==75
                finally: os.environ.pop('PIDG_CAPTURE_INITIAL_ONLY', None)
            from reproduction.scripts.training_checkpoint import load_state
            validate_initial_state(load_state(initial/'checkpoints/resume.pt'))
            control, iso = [initialize(n, args.smoke) for n in NAMES]
            for d in (control, iso): clone_zero(initial, d)
            receipt = {'iteration': 0, 'optimizer_updates': 0, 'pretrained_model_used': False,
                'sha256': engine.sha256(initial/'checkpoints/resume.pt'),
                'initialization_run': str(initial), 'destinations': [str(control), str(iso)],
                'shared_random_geometry': str(base.DATA/'training/points3d.ply')}
            engine.write_json(REPORTS/('smoke_initialization.json' if args.smoke else 'initialization.json'), receipt)
            target = 100 if args.smoke else 50000
            if not args.smoke:
                state['status'] = 'training_from_scratch'; engine.write_json(RUNS/'controller.json', state)
            # Start the requested iso model first; the equally initialized control follows.
            for d in (iso, control):
                if args.smoke and d == iso and step(d)<20:
                    assert base.train(d, repo, deadline, True, limit=20)==75
                if step(d)<target:
                    if base.train(d, repo, deadline, args.smoke)==75: raise SystemExit(75)
                state['completed'].append(d.parent.name)
                engine.write_json(RUNS/'controller.json', state)
            if args.smoke:
                gradients = [json.loads(s) for s in (iso/'isotropic_gradient.jsonl').read_text().splitlines()]
                if not gradients or not any(g['scale_gradient_l1']>0 for g in gradients): raise ValueError('No isotropic gradient')
                base.render(iso, repo, deadline, split='validation', tags=['best'], smoke=True)
                split = engine.read_json(ROOT/'experiments/kinematic_resplit/split_audit.json')['frame_ids']
                for d in (control, iso):
                    saved = load_state(d/'checkpoints/resume.pt')
                    if {int(n) for n in saved['train_camera_names']} != set(split['train']): raise ValueError('Training membership changed')
                    if {int(n) for n in saved['test_camera_names']} != set(split['val']): raise ValueError('Held-out membership changed')
                engine.write_json(REPORTS/'preflight.json', {'status': 'passed', 'flags_off_steps': 20,
                    'random_initial_state_iteration': 0, 'optimizer_updates_at_initialization': 0,
                    'identical_initial_checkpoint_sha256': receipt['sha256'], 'control_iso_steps': 100,
                    'iso_full_resume_from': 20, 'iso_scale_gradients': gradients,
                    'validation_png_parity': True, 'train_validation_membership_verified': True,
                    'completed_utc': engine.now(), 'source_freeze_sha256': engine.sha256(REPORTS/'source_freeze.json')})
                state['status'] = 'preflight_passed'
            else:
                selections = {d.parent.name: engine.read_json(d/'metric_best.json') for d in (control, iso)}
                lock = RUNS/'selection_lock.json'
                if lock.exists():
                    if engine.read_json(lock)['validation_selections'] != selections: raise ValueError('Locked selections changed')
                else: engine.write_json(lock, {'locked_utc': engine.now(), 'validation_selections': selections, 'test_tags': TAGS})
                state['status'] = 'final_test'; engine.write_json(RUNS/'controller.json', state)
                for d in (iso, control):
                    base.render(d, repo, deadline)
                    base.pairs(d)
                state['status'] = 'completed'
            assert_frozen(repo)
        except SystemExit as error:
            state['status'] = 'saved_stopped' if error.code==75 else 'failed'
            if error.code != 75: raise
        except Exception as error:
            state.update(status='failed', error=str(error)); raise
        finally:
            state['finished_utc'] = engine.now()
            engine.write_json(RUNS/'controller.json', state)
            stop.set(); watcher.join(timeout=6); report()


if __name__ == '__main__': main()
