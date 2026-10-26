"""Matched 50k V3 continuations: iso=0 versus iso=1e-3; test locked until end."""
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
from variants.v3_isotropic.prepare import prepare

HERE = Path(__file__).resolve().parent
RUNS = ROOT/'runs/pidg_v3_isotropic'
REPORTS = ROOT/'experiments/v3_isotropic'
PARENT = ROOT/'runs/pidg_kinematic_resplit/V3_frame_time/rubber_duck'
NAMES = ['V3_continue_control', 'V3_iso_1e3']
TAGS = base.TAGS


def configuration(name, smoke=False, off=False):
    config = engine.read_json(PARENT/'metric_config.json')
    config.update(parent_run='' if off else str(PARENT), enabled=not off,
        isotropic_weight=.001 if name == NAMES[1] and not off else 0.,
        isotropic_scope='rendered', isotropic_eps=1e-8, isotropic_start=20 if smoke else 1000,
        kinematic_weight=0., engineering_smoke=smoke)
    text = (PARENT/'model_config.py').read_text(encoding='utf-8')
    if smoke: text += '\nposition_lr_max_steps = 100\ndeform_lr_max_steps = 100\n'
    return config, text


def branch_config(a, b):
    a, b = dict(a), dict(b)
    if a.pop('isotropic_weight') != 0. or b.pop('isotropic_weight') != .001 or a != b:
        raise ValueError('The only branch change must be isotropic_weight: 0 -> 0.001')


def initialize(name, smoke=False, off=False):
    directory = RUNS/('_smoke' if smoke else '')/name/'rubber_duck'
    config, text = configuration(name, smoke, off)
    if (directory/'reproduction.json').exists():
        if engine.read_json(directory/'metric_config.json') != config or (directory/'model_config.py').read_text() != text:
            raise ValueError('Existing V3+iso run changed')
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    engine.write_json(directory/'metric_config.json', config)
    engine.write_text_atomic(directory/'model_config.py', text)
    if not off:
        for folder in ('point_cloud', 'deform'):
            shutil.copytree(PARENT/folder/'iteration_best', directory/folder/'iteration_best')
    engine.write_json(directory/'reproduction.json', {'id': name, 'status': 'initialized',
        'parent': str(PARENT) if not off else '', 'warm_start': not off,
        'target_additional_iterations': 100 if smoke else 50000, 'seed': 0,
        'train': 120, 'validation': 30, 'test': 50, 'test_selection': False,
        'resolution_width': 400 if smoke else 1600, 'tuning': config,
        'source_freeze_sha256': engine.sha256(REPORTS/'source_freeze.json'), 'commands': []})
    return directory


def freeze(repo):
    files = [*HERE.glob('*.py'), ROOT/'variants/isotropic_tuning/model.py', ROOT/'variants/isotropic_tuning/runtime.py',
             PARENT/'metric_config.json', PARENT/'model_config.py', PARENT/'metric_best.json',
             PARENT/'results.json', PARENT/'metric_history.jsonl']
    for folder in ('point_cloud', 'deform'):
        files += [p for p in (PARENT/folder/'iteration_best').rglob('*') if p.is_file()]
    return {'parent_resplit_source_data': base.freeze(base.HERE/'code'), 'code': engine.runtime_hashes(repo),
            'inputs': {str(p.relative_to(ROOT)): engine.sha256(p) for p in files}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json') != freeze(repo):
        raise ValueError('V3+iso source, data or parent weights changed')


def wire(repo):
    # Only process-local coordinator globals change; frozen parent files stay intact.
    base.RUNS, base.REPORTS, base.assert_frozen, base.branch_config = RUNS, REPORTS, assert_frozen, branch_config
    engine.RUNS, engine.REPORTS, engine.repo_global, engine.assert_frozen = RUNS, REPORTS, repo, assert_frozen


def report():
    state = engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active = engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    rows = []
    for name in NAMES:
        p = RUNS/name/'rubber_duck'
        hist = base.history(p)
        rows.append({'id': name, 'iteration': base.progress(p), 'history': hist,
                     'test': engine.read_json(p/'results.json') if (p/'results.json').exists() else None})
    old_test = engine.read_json(PARENT/'results.json')['ours_50000']
    paired = None
    if all(r['test'] for r in rows):
        a, b = [r['test']['ours_50000'] for r in rows]
        views = [engine.read_json(RUNS/n/'rubber_duck/per_view.json')['ours_50000'] for n in NAMES]
        paired = {'iso_minus_equal_budget_control': {k: b[k]-a[k] for k in engine.METRICS},
            'improved_frames': {k: sum(engine.metric_score(views[1][k][n]-views[0][k][n], k)>0 for n in views[0][k])
                                for k in engine.METRICS}, 'test_frames': 50}
    result = {'updated_utc': engine.now(), 'controller': state, 'active': active,
        'protocol': 'V3 parent; additional50k; 120 train/30 validation/50 test; shared frame clock; iso=0 vs .001',
        'parent_test_reference': old_test, 'rows': rows, 'paired_final50k': paired,
        'limitations': ['historical official-test exposure', 'single scene and seed',
                        '50k extra training for both arms; compare iso versus continued control to isolate regularizer',
                        'uniform frame cadence assumed; no ground-truth physical validation']}
    engine.write_json(REPORTS/'results.json', result)
    lines = ['# V3 + isotropic regularization', '', result['protocol'], '',
        'Status: '+state.get('status', 'preparing'), 'Active: '+active.get('id', '-')+' / '+active.get('phase', '-'), '',
        'Both arms inherit the same completed V3 validation-selected weights. The first1000 steps share a full checkpoint; iso starts at1001.',
        'Validation alone selects the four best snapshots. After both arms finish, lock selections and evaluate final50k plus four validation-selected snapshots on test.', '',
        '| Model | Extra iter | Latest validation PSNR |', '|---|---:|---:|']
    for r in rows:
        value = r['history'][-1]['metrics']['PSNR'] if r['history'] else '-'
        lines.append(f"| {r['id']} | {r['iteration']}/50000 | {value} |")
    lines += ['', '## Fixed-step final test results', '',
        '| Model | PSNR | SSIM | LPIPS | MS-SSIM |', '|---|---:|---:|---:|---:|',
        '| Previous V3 (before extra training) | '+' | '.join(f'{old_test[k]:.7f}' for k in engine.METRICS)+' |']
    for r in rows:
        if r['test']:
            m = r['test']['ours_50000']
            lines.append('| '+r['id']+' | '+' | '.join(f'{m[k]:.7f}' for k in engine.METRICS)+' |')
    if paired:
        lines += ['', '| Metric | Iso minus continued control | Improved frames / 50 |', '|---|---:|---:|']
        for k in engine.METRICS:
            lines.append(f"| {k} | {paired['iso_minus_equal_budget_control'][k]:+.7f} | {paired['improved_frames'][k]} |")
    lines += ['', '## Validation-selected snapshots: final test metrics', '',
              '| Model | Validation objective | Extra iter | PSNR | SSIM | LPIPS | MS-SSIM |', '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if not r['test']: continue
        for objective, best in engine.read_json(RUNS/r['id']/'rubber_duck/metric_best.json').items():
            m = r['test']['ours_'+best['snapshot']]
            lines.append('| '+' | '.join([r['id'], objective, str(best['iteration']),
                *[f'{m[k]:.7f}' for k in engine.METRICS]])+' |')
    lines += ['', 'The previous V3 test set has historical exposure. This prospective validation-only protocol cannot erase it.',
              '[5-second chart monitor](monitor.html) | [Plan](plan.json) | [GPU preflight](preflight.json)', '',
              'GT/render pairs absolute directory: '+str(RUNS/'render_comparisons'), '', 'Updated UTC: '+result['updated_utc']]
    engine.write_text_atomic(REPORTS/'results.md', '\n'.join(lines)+'\n')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>V3 + iso</title><style>body{font:16px system-ui;background:#111c2c;color:#e5edf8;margin:30px}section,svg{background:#203047;padding:15px;margin:10px 0}progress{width:80%}a{color:#82e6bd}svg{max-width:100%}</style><h1>V3 + isotropic regularization</h1><p>Refresh 5s; 120 train / 30 validation / 50 test. Charts use validation only.</p><p>'+html.escape(state.get('status', 'preparing')+' | '+active.get('id', '-')+' | '+active.get('phase', '-'))+'</p>'
    for r in rows:
        page += f"<section><b>{r['id']}</b><h2>{r['iteration']:,} / 50,000 extra iter</h2><progress value='{r['iteration']}' max='50000'></progress></section>"
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
    page += '<p><a href="results.md">Full results and limitations</a></p></html>'
    engine.write_text_atomic(REPORTS/'monitor.html', page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--hours', type=float, default=24)
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    parent_repo = base.prepare()
    # Validate the old study before redirecting any coordinator paths.
    if engine.read_json(ROOT/'experiments/kinematic_resplit/source_freeze.json') != base.freeze(parent_repo):
        raise ValueError('Frozen parent study changed')
    repo = prepare()
    frozen = freeze(repo)
    if (REPORTS/'source_freeze.json').exists(): assert_frozen(repo)
    else: engine.write_json(REPORTS/'source_freeze.json', frozen)
    for name in NAMES:
        config, text = configuration(name)
        engine.write_json(REPORTS/'configs'/(name+'.json'), config)
        engine.write_text_atomic(REPORTS/'configs'/(name+'.py'), text)
    engine.write_json(REPORTS/'plan.json', {'parent': str(PARENT), 'parent_selected_by': 'validation PSNR',
        'arms': dict(zip(NAMES, [0., .001])), 'additional_steps': 50000, 'shared_full_state_prefix': 1000,
        'iso_first_active_step': 1001, 'iso_scope': 'rendered axis magnitudes abs(exp(log_scales)+d_scaling)',
        'iso_loss': 'mean_i mean_axis abs(abs(s_ia)/mean_axis(abs(s_i))-1); robust normalized implementation',
        'kinematic_weight': 0., 'train_validation_test': [120, 30, 50], 'selection': 'validation only',
        'final_test_tags': TAGS, 'resolution': [1600, 900], 'seed': 0, 'budget_hours': args.hours,
        'test_history': 'Official test set has prior tuning exposure; disclosed exploratory comparison'})
    wire(repo)
    deadline = datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state = {'pid': os.getpid(), 'status': 'preflight' if args.smoke else 'training',
             'started_utc': engine.now(), 'deadline_utc': deadline.isoformat(), 'completed': []}
    stop = threading.Event()
    def monitor():
        while not stop.wait(5):
            try: report()
            except (OSError, ValueError, KeyError): pass
    def step(d):
        return engine.read_json(d/'checkpoints/resume.json')['iteration'] if (d/'checkpoints/resume.json').exists() else 0
    with engine.lock():
        if (RUNS/'stop_queue.request').exists(): raise RuntimeError('Pause request remains')
        engine.write_json(RUNS/'controller.json', state); report()
        watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
        try:
            if args.smoke:
                off = initialize('flags_off', True, True)
                if step(off)<20: assert base.train(off, repo, deadline, smoke=True, off=True) == 0
            else:
                if engine.read_json(REPORTS/'preflight.json')['status'] != 'passed': raise ValueError('Preflight required')
            control, iso = [initialize(n, args.smoke) for n in NAMES]
            target, boundary = (100, 20) if args.smoke else (50000, 1000)
            if not (iso/'branch.json').exists():
                if step(control)<boundary:
                    assert base.train(control, repo, deadline, args.smoke, limit=boundary)==75
                base.branch(control, iso, boundary)
                receipt = engine.read_json(iso/'branch.json')
                receipt['only_change'] = 'isotropic_weight: 0 -> 0.001'
                engine.write_json(iso/'branch.json', receipt)
            for d in (control, iso):
                if step(d)<target:
                    if base.train(d, repo, deadline, args.smoke)==75: raise SystemExit(75)
                state['completed'].append(d.parent.name)
                engine.write_json(RUNS/'controller.json', state)
            if args.smoke:
                grads = [json.loads(s) for s in (iso/'isotropic_gradient.jsonl').read_text().splitlines()]
                if not grads or not all(g['scale_gradient_l1']>0 for g in grads): raise ValueError('Missing iso gradient')
                base.render(iso, repo, deadline, split='validation', tags=['best'], smoke=True)
                from reproduction.scripts.training_checkpoint import load_state
                split = engine.read_json(ROOT/'experiments/kinematic_resplit/split_audit.json')['frame_ids']
                for d in (control, iso):
                    saved = load_state(d/'checkpoints/resume.pt')
                    if {int(n) for n in saved['train_camera_names']} != set(split['train']): raise ValueError('Training membership changed')
                    if {int(n) for n in saved['test_camera_names']} != set(split['val']): raise ValueError('Final-test RGB entered training loader')
                engine.write_json(REPORTS/'preflight.json', {'status': 'passed', 'flags_off_steps': 20,
                    'control_iso_steps': 100, 'full_state_branch_at': 20, 'finite_nonzero_iso_scale_gradients': grads,
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
                for d in (control, iso):
                    base.render(d, repo, deadline)
                    base.pairs(d)
                state['status'] = 'completed'
            assert_frozen(repo)
        except SystemExit as error:
            state['status'] = 'saved_stopped' if error.code == 75 else 'failed'
            if error.code != 75: raise
        except Exception as error:
            state.update(status='failed', error=str(error)); raise
        finally:
            state['finished_utc'] = engine.now()
            engine.write_json(RUNS/'controller.json', state)
            stop.set(); watcher.join(timeout=6); report()


if __name__ == '__main__': main()
