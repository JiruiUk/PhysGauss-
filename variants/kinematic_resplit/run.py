"""Train-only initialization, matched full-state branches, locked final test."""
import argparse
from datetime import datetime, timedelta, timezone
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.metric_tuning import run as engine
from variants.kinematic_tuning import run as old
from variants.kinematic_resplit.prepare import prepare
from variants.kinematic_resplit.data import prepare as prepare_data

HERE = Path(__file__).resolve().parent
RUNS = ROOT/'runs/pidg_kinematic_resplit'
REPORTS = ROOT/'experiments/kinematic_resplit'
DATA = ROOT/'data/derived/kinematic_resplit'
NAMES = ['P0_train_only', 'V3_frame_time', 'V4_frame_time_kinematic']
TAGS = list(dict.fromkeys(engine.TAGS.values()))+['50000']


def branch_config(source, destination):
    a, b = dict(source), dict(destination)
    if a.pop('kinematic_weight') != 0. or b.pop('kinematic_weight') != .001 or a != b:
        raise ValueError('Only the predeclared kinematic weight 0 -> .001 may differ')


def configure(name, iterations=50000, smoke=False, off=False):
    preset = dict(old.PRESETS[2], id=name, kinematic_weight=.001 if name.startswith('V4') else 0.)
    config, text = old.configuration(preset, iterations)
    config['parent_run'] = '' if name.startswith('P0') or off else str(RUNS/('_smoke/P0_train_only' if smoke else 'P0_train_only')/'rubber_duck')
    config.update(kinematic_start=20 if smoke else 1000, kinematic_ramp=1 if smoke else 1000,
                  audit_gradients=smoke, audit_parameters=smoke, enabled=not off)
    if name.startswith('P0') or off:
        # Reproduce the original baseline optimization recipe on train-only data.
        text = (engine.SOURCE/'arguments/pidg/rubber_dack.py').read_text(encoding='utf-8')
        text += '\nnetwork_args = dict(network_args, width=256, depth=0, metric_residual_width=0)\n'
        config.update(parent_run='', perceptual_weight=0., shared_physics_time=False,
                      timeline='official_metadata', kinematic_weight=0.)
    if smoke or off: text += '\nwarm_up = 0\ndensify_until_iter = 0\n'
    return config, text


def initialize(name, smoke=False, off=False):
    target = RUNS/('_smoke' if smoke else '')/name/'rubber_duck'
    config, text = configure(name, 100 if smoke else 50000, smoke, off)
    if (target/'reproduction.json').exists():
        if engine.read_json(target/'metric_config.json') != config or (target/'model_config.py').read_text() != text:
            raise ValueError('Existing run config changed')
        return target
    target.mkdir(parents=True, exist_ok=True)
    engine.write_json(target/'metric_config.json', config)
    engine.write_text_atomic(target/'model_config.py', text)
    if config['parent_run']:
        parent = Path(config['parent_run'])
        for folder in ('point_cloud', 'deform'):
            shutil.copytree(parent/folder/'iteration_best', target/folder/'iteration_best')
    engine.write_json(target/'reproduction.json', {'id': name, 'status': 'initialized',
        'parent': config['parent_run'], 'warm_start': bool(config['parent_run']),
        'resolution_width': 400 if smoke else 1600, 'target_iterations': 100 if smoke else 50000,
        'train': 120, 'validation': 30, 'test': 50, 'selection': 'validation only',
        'source_freeze_sha256': engine.sha256(REPORTS/'source_freeze.json'), 'commands': []})
    return target


def freeze(repo):
    paths = [*HERE.glob('*.py'), ROOT/'variants/metric_tuning/runtime.py', ROOT/'variants/metric_tuning/model.py',
        ROOT/'variants/metric_tuning/checkpoint.py', ROOT/'variants/metric_tuning/run.py',
        ROOT/'variants/kinematic_tuning/runtime.py', ROOT/'variants/kinematic_tuning/model.py',
        ROOT/'variants/kinematic_tuning/run.py', ROOT/'reproduction/scripts/training_checkpoint.py',
        ROOT/'scripts/pidg_variant_entrypoint.py']
    paths += list(DATA.glob('*.json'))
    for mode in ('training', 'final_test'):
        paths += list((DATA/mode).glob('*.json'))+[DATA/mode/'points3d.ply']
    return {'code': engine.runtime_hashes(repo), 'parent_code': engine.runtime_hashes(old.HERE/'code'),
            'inputs': {str(p.relative_to(ROOT)): engine.sha256(p) for p in paths}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json') != freeze(repo):
        raise ValueError('Experiment source or split changed')


def train(directory, repo, deadline, smoke=False, limit=0, off=False):
    assert_frozen(repo)
    iterations = 20 if off else 100 if smoke else 50000
    tests = [100] if off else [20, 100] if smoke else list(range(1000, 50001, 1000))
    command = [*engine.command_prefix(repo), 'train_pidg.py', '-s', str(DATA/'training'), '-m', str(directory),
        '--conf', str(directory/'model_config.py'), '--load2gpu_on_the_fly', '--resolution', '400' if smoke else '1600',
        '--iterations', str(iterations), '--checkpoint_interval', '20' if smoke else '1000',
        '--save_image_dir', str(directory/'diagnostics'), '--save_iterations', str(iterations),
        '--test_iterations', *map(str, tests)]
    checkpoint = directory/'checkpoints/resume.pt'
    if checkpoint.exists():
        if engine.sha256(checkpoint) != engine.read_json(checkpoint.with_suffix('.json'))['sha256']:
            raise ValueError('Invalid full checkpoint')
        command += ['--resume_checkpoint', str(checkpoint)]
    env = engine.env_for(directory)
    if limit: env['PIDG_TUNING_STOP_AFTER'] = str(limit)
    return engine.execute(command, directory, 'flags_off' if off else 'long', env, deadline)


def branch(source, destination, boundary):
    if (destination/'branch.json').exists(): return
    if (destination/'checkpoints/resume.pt').exists(): raise ValueError('Unreceipted branch state exists')
    a, b = [engine.read_json(d/'metric_config.json') for d in (source, destination)]
    branch_config(a, b)
    manifest = engine.read_json(source/'checkpoints/resume.json')
    if manifest['iteration'] != boundary: raise ValueError('Wrong common branch boundary')
    for folder in ('point_cloud', 'deform', 'checkpoints'):
        shutil.copytree(source/folder, destination/folder, dirs_exist_ok=True)
    for name in ('metric_best.json', 'metric_history.jsonl'):
        shutil.copyfile(source/name, destination/name)
    sidecar = destination/'checkpoints/resume.metrics.json'
    state = engine.read_json(sidecar)
    if state['config'] != a: raise ValueError('Unexpected common checkpoint config')
    state['config'] = b  # Explicit branch receipt, tensor/optimizer/RNG bytes are unchanged.
    engine.write_json(sidecar, state)
    digest = engine.sha256(destination/'checkpoints/resume.pt')
    if digest != manifest['sha256']: raise ValueError('Full state was not copied identically')
    engine.write_json(destination/'branch.json', {'source': str(source), 'destination': str(destination), 'iteration': boundary,
        'identical_full_checkpoint_sha256': digest, 'source_config': a, 'branch_config': b,
        'only_change': 'kinematic_weight: 0 -> 0.001', 'created_utc': engine.now()})


def render(directory, repo, deadline, split='test', tags=None, smoke=False):
    tags = tags or TAGS
    source = DATA/('final_test' if split == 'test' else 'training')
    count = 50 if split == 'test' else 30
    # Render's legacy output folder is always test; keep validation exports separate.
    if split == 'test' and not (RUNS/'selection_lock.json').exists(): raise ValueError('Final test is still locked')
    for tag in tags:
        raw = directory/'test'/('ours_'+tag)
        receipt = raw/'render_complete.json'
        if receipt.exists() and engine.read_json(receipt).get('split') == split: continue
        command = [*engine.command_prefix(repo), 'render.py', '-s', str(source), '-m', str(directory),
            '--conf', str(directory/'model_config.py'), '--load2gpu_on_the_fly', '--resolution', '400' if smoke else '1600',
            '--iteration', tag, '--skip_train']
        engine.execute(command, directory, 'render_'+split+'_'+tag, engine.env_for(directory), deadline)
        expected = {f'{i:05d}.png' for i in range(count)}
        if not all({p.name for p in (raw/k).glob('*.png')} == expected for k in ('renders', 'gt')):
            raise ValueError('Incomplete PNG export')
        engine.write_json(receipt, {'split': split, 'images': count})
    started = time.time()
    engine.execute([*engine.command_prefix(repo), 'metrics.py', '-m', str(directory)], directory,
                   'metrics_'+split, engine.env_for(directory), deadline)
    if not (directory/'results.json').exists() or (directory/'results.json').stat().st_mtime < started:
        raise ValueError('Metrics command did not produce a fresh result')
    verify_metrics(engine.read_json(directory/'results.json'), engine.read_json(directory/'per_view.json'), tags, count)
    if split == 'validation':
        results = engine.read_json(directory/'results.json')
        best = engine.read_json(directory/'metric_best.json')
        for tag in tags:
            reference = next(v['metrics'] for v in best.values() if v['snapshot'] == tag)
            if any(abs(results['ours_'+tag][k]-reference[k]) > 1e-4 for k in engine.METRICS):
                raise ValueError('Validation PNG parity failure')


def verify_metrics(results, per_view, tags, count):
    expected = {f'{i:05d}.png' for i in range(count)}
    for tag in tags:
        key = 'ours_'+tag
        for metric in engine.METRICS:
            mean = results[key][metric]
            values = per_view[key][metric]
            if set(values) != expected or not all(math.isfinite(v) for v in [mean, *values.values()]):
                raise ValueError('Incomplete or nonfinite per-frame metrics')
            if abs(mean-math.fsum(values.values())/count) > 1e-5:
                raise ValueError('Mean metric does not match per-frame data')


def pairs(directory):
    # Reuse the unchanged pixel-identity-checked, 50-view pair exporter.
    previous = old.RUNS
    try:
        old.RUNS = RUNS
        old.export_pairs(directory)
    finally: old.RUNS = previous


def history(directory):
    return old.history(directory)


def progress(directory):
    text = engine.tail(directory/'train.log', 100000)
    matches = re.findall(r'Training progress[^\r\n]*?\|\s*(\d+)/50000', text)
    candidates = [int(n) for n in matches]
    path = directory/'checkpoints/resume.json'
    if path.exists(): candidates.append(engine.read_json(path)['iteration'])
    candidates += [r['iteration'] for r in history(directory)]
    return max(candidates, default=0)


def report():
    state = engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active = engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    rows = []
    for name in NAMES:
        directory = RUNS/name/'rubber_duck'
        h = history(directory)
        rows.append({'id': name, 'iteration': progress(directory), 'target': 50000,
            'validation': h[-1] if h else None,
            'test': engine.read_json(directory/'results.json') if (directory/'results.json').exists() else None})
    comparison = []
    for row in rows[1:]:
        old_run = ROOT/'runs/pidg_kinematic_tuning'/row['id']/'rubber_duck/results.json'
        if row['test']:
            before, after = engine.read_json(old_run)['ours_50000'], row['test']['ours_50000']
            comparison.append({'model': row['id'], 'previous_final50k': before, 'resplit_final50k': after,
                'delta': {k: after[k]-before[k] for k in engine.METRICS}})
    paired = None
    if all(r['test'] for r in rows[1:]):
        a, b = [r['test']['ours_50000'] for r in rows[1:]]
        views = [engine.read_json(RUNS/n/'rubber_duck/per_view.json')['ours_50000'] for n in NAMES[1:]]
        paired = {'V4_minus_V3': {k: b[k]-a[k] for k in engine.METRICS},
            'improved_frame_count': {k: sum(engine.metric_score(views[1][k][name]-views[0][k][name], k) > 0
                for name in views[0][k]) for k in engine.METRICS}, 'test_frames': 50}
    result = {'updated_utc': engine.now(), 'controller': state, 'active': active, 'rows': rows,
        'comparison': comparison, 'paired_final50k': paired,
        'protocol': '120 train / 30 validation / unchanged 50 test; global frame clock',
        'limitations': ['previous test exposure persists', 'single scene/seed', 'train count and clean initialization differ from previous runs',
                        'uniform frame cadence assumed', 'CUDA training may diverge after identical full-state branching']}
    engine.write_json(REPORTS/'results.json', result)
    lines = ['# V3 / V4 time-disjoint retraining', '', result['protocol'], '',
        'Train and validation loader views contain no final-test RGB. Validation selects snapshots; test is evaluated only after both models finish and selections are locked.',
        'The prior official test set has historical tuning exposure. This is a prospectively separated re-evaluation, not a new untouched benchmark.', '',
        'Status: '+state.get('status', 'preparing'), 'Active: '+active.get('id', '-')+' / '+active.get('phase', '-'), '',
        '| Stage | Iteration | Latest validation PSNR |', '|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['id']} | {r['iteration']}/50000 | {r['validation']['metrics']['PSNR'] if r['validation'] else '-'} |")
    lines += ['', '## Final test comparison (same 50 RGB, poses, and frame-index times)', '',
        '| Model | Metric | Previous final50k | Resplit final50k | Change |', '|---|---|---:|---:|---:|']
    for row in comparison:
        for k in engine.METRICS:
            lines.append(f"| {row['model']} | {k} | {row['previous_final50k'][k]:.7f} | {row['resplit_final50k'][k]:.7f} | {row['delta'][k]:+.7f} |")
    if paired:
        lines += ['', '## Controlled new V4 versus new V3 at final50k', '',
                  '| Metric | V4 minus V3 | Improved frames / 50 |', '|---|---:|---:|']
        for k in engine.METRICS:
            lines.append(f"| {k} | {paired['V4_minus_V3'][k]:+.7f} | {paired['improved_frame_count'][k]} |")
    lines += ['', '## Validation-selected snapshots: final test scores', '',
              '| Model | Validation objective | Selected iter | PSNR | SSIM | LPIPS | MS-SSIM |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in rows[1:]:
        if not r['test']: continue
        best = engine.read_json(RUNS/r['id']/'rubber_duck/metric_best.json')
        for objective, selected in best.items():
            m = r['test']['ours_'+selected['snapshot']]
            lines.append('| '+' | '.join([r['id'], objective, str(selected['iteration']),
                *[f'{m[k]:.7f}' for k in engine.METRICS]])+' |')
    lines += ['', 'Raw before/after changes include fewer training images and a newly trained parent. They do not isolate the effect of validation separation.',
              'The controlled method comparison is new V4 versus new V3, including per-frame metrics.', '',
              '[5-second graphical monitor](monitor.html) | [Split audit](split_audit.json) | [Plan](plan.json)', '',
              'GT/render pairs: '+str(RUNS/'render_comparisons'), '', 'Updated UTC: '+result['updated_utc']]
    engine.write_text_atomic(REPORTS/'results.md', '\n'.join(lines)+'\n')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>V3/V4 clean split</title><style>body{font:16px system-ui;background:#111c2c;color:#e5edf8;margin:30px}section{background:#203047;padding:18px;margin:12px 0}progress{width:80%}a{color:#81d7bd}svg{background:#203047;max-width:100%}</style><h1>V3 / V4: time-disjoint retraining</h1><p>120 train / 30 validation / 50 test. Refresh: 5s. Curves show validation only.</p><p>'+html.escape(state.get('status', 'preparing')+' | '+active.get('id', '-')+' | '+active.get('phase', '-'))+'</p>'
    for r in rows:
        page += f"<section><b>{r['id']}</b><h2>{r['iteration']:,} / 50,000 iter</h2><progress max='50000' value='{r['iteration']}'></progress></section>"
    colors = ['#fac078', '#8fd3ff', '#82e6bd']
    for key in engine.METRICS:
        curves = [(name, history(RUNS/name/'rubber_duck'), colors[i]) for i, name in enumerate(NAMES)]
        values = [r['metrics'][key] for _, h, _ in curves for r in h]
        if not values: continue
        lo, hi = min(values), max(values)
        span = max(hi-lo, 1e-6)
        page += f'<h2>Validation {key}</h2><svg viewBox="0 0 900 260" width="900"><text x="8" y="22" fill="white">{hi:.5f}</text><text x="8" y="220" fill="white">{lo:.5f}</text>'
        for i, (name, h, color) in enumerate(curves):
            points = ' '.join(f"{80+790*r['iteration']/50000:.1f},{220-190*(r['metrics'][key]-lo)/span:.1f}" for r in h)
            page += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{80+i*270}" y="250" fill="{color}" font-size="12">{name}</text>'
        page += '</svg>'
    page += '<p><a href="results.md">Results and limitations</a></p></html>'
    engine.write_text_atomic(REPORTS/'monitor.html', page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--hours', type=float, default=24)
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    audit = prepare_data(ROOT/'data/PIDG/rubber_duck', DATA)
    engine.write_json(REPORTS/'split_audit.json', audit)
    repo = prepare()
    frozen = freeze(repo)
    if (REPORTS/'source_freeze.json').exists(): assert_frozen(repo)
    else: engine.write_json(REPORTS/'source_freeze.json', frozen)
    engine.RUNS, engine.REPORTS, engine.repo_global = RUNS, REPORTS, repo
    engine.assert_frozen = assert_frozen
    engine.write_json(REPORTS/'plan.json', {'counts': audit['counts'], 'clock': audit['time_rule'],
        'P0': '50k original PIDG recipe, random initialization, 120 train, validation selection, no old weights',
        'V3_V4': '50k additional T3 refinement each; common first1000 full-state checkpoint; only lambda_kin differs',
        'loss': 'unchanged V3/V4: shared time, Huber(v, d[R(q)X+b]/dtau), weight 0 / .001',
        'selection': 'four objectives on 30 validation images, every1000; final50k also retained',
        'final_test': 'unchanged official50 images; locked until both arms finish; all five predeclared snapshots',
        'prior_test_exposure': audit['prior_exposure'], 'seed': 0, 'resolution': [1600, 900], 'budget_hours': args.hours})
    deadline = datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state = {'pid': os.getpid(), 'status': 'preflight' if args.smoke else 'training',
             'started_utc': engine.now(), 'deadline_utc': deadline.isoformat(), 'completed': []}
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
            if not args.smoke:
                if engine.read_json(REPORTS/'preflight.json')['status'] != 'passed': raise ValueError('Preflight required')
            else:
                disabled = initialize('flags_off', smoke=True, off=True)
                if not (disabled/'checkpoints/resume.json').exists() or engine.read_json(disabled/'checkpoints/resume.json')['iteration'] < 20:
                    assert train(disabled, repo, deadline, smoke=True, off=True) == 0
            parent = initialize(NAMES[0], args.smoke)
            target = 100 if args.smoke else 50000
            step = lambda d: engine.read_json(d/'checkpoints/resume.json')['iteration'] if (d/'checkpoints/resume.json').exists() else 0
            if step(parent) < target:
                if train(parent, repo, deadline, args.smoke) == 75: raise SystemExit(75)
            state['completed'].append(NAMES[0])
            engine.write_json(RUNS/'controller.json', state)
            v3, v4 = [initialize(n, args.smoke) for n in NAMES[1:]]
            boundary = 20 if args.smoke else 1000
            if not (v4/'branch.json').exists():
                if step(v3) < boundary:
                    assert train(v3, repo, deadline, args.smoke, limit=boundary) == 75
                branch(v3, v4, boundary)
            for d in (v3, v4):
                if step(d) < target:
                    if train(d, repo, deadline, args.smoke) == 75: raise SystemExit(75)
                state['completed'].append(d.parent.name)
                engine.write_json(RUNS/'controller.json', state)
            if args.smoke:
                render(v4, repo, deadline, split='validation', tags=['best'], smoke=True)
                grads = [json.loads(s) for s in (v4/'kinematic_parameter_gradient.jsonl').read_text().splitlines()]
                if {g['branch'] for g in grads} != {'grid', 'deformation', 'velocity'} or not all(g['finite'] and g['gradient_l1'] > 0 for g in grads):
                    raise ValueError('Invalid kinematic parameter gradients')
                for d in (parent, v3, v4):
                    from reproduction.scripts.training_checkpoint import load_state
                    saved = load_state(d/'checkpoints/resume.pt')
                    expected = audit['frame_ids']
                    if {int(n) for n in saved['train_camera_names']} != set(expected['train']): raise ValueError('Training cameras leaked')
                    if {int(n) for n in saved['test_camera_names']} != set(expected['val']): raise ValueError('Validation camera mismatch')
                engine.write_json(REPORTS/'preflight.json', {'status': 'passed', 'completed_utc': engine.now(),
                    'flags_off_steps': 20, 'scratch_parent_steps': 100, 'V3_V4_steps': 100,
                    'identical_full_state_branch_at': boundary, 'actual_parameter_gradients': grads,
                    'checkpoint_train_val_membership_verified': True, 'test_rgb_absent_from_training_view': True,
                    'validation_PNG_metric_parity': True, 'tests': 'tests/test_kinematic_resplit.py and existing kinematic tests'})
                state['status'] = 'preflight_passed'
            else:
                selections = {d.parent.name: engine.read_json(d/'metric_best.json') for d in (v3, v4)}
                selection_path = RUNS/'selection_lock.json'
                if selection_path.exists():
                    if engine.read_json(selection_path)['validation_selections'] != selections: raise ValueError('Locked selections changed')
                else:
                    engine.write_json(selection_path, {'locked_utc': engine.now(), 'validation_selections': selections,
                        'test_tags': TAGS, 'test_rgb_never_used_by_this_training': True})
                state['status'] = 'final_test'; engine.write_json(RUNS/'controller.json', state)
                for d in (v3, v4):
                    render(d, repo, deadline)
                    pairs(d)
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
            stop.set(); thread.join(timeout=6); report()


if __name__ == '__main__': main()
