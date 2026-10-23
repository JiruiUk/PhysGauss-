"""Run isolated T3/isotropy controls or bounded GPU validation, preserving old runs."""
import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shutil
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.metric_tuning import run as engine
from variants.isotropic_tuning.prepare import prepare

HERE = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/pidg_isotropic_tuning'
REPORTS = ROOT / 'experiments/isotropic_tuning'
PRESETS = [
    {'id': 'I0_T3_control', 'isotropic_weight': 0.},
    {'id': 'I1_iso_1e4', 'isotropic_weight': 1e-4},
    {'id': 'I2_iso_1e3', 'isotropic_weight': 1e-3},
    {'id': 'I3_iso_1e2', 'isotropic_weight': 1e-2},
]
T3 = {'parent': 'W256_D0', 'pixel_loss': 'l1', 'ssim': .2, 'reg': .5, 'perceptual': .02}
capacity_references = engine.references


def reference_rows():
    rows = capacity_references()
    directory = ROOT / 'runs/pidg_metric_tuning/T3_baseline_perceptual/rubber_duck'
    if (directory / 'results.json').exists():
        scores = engine.read_json(directory / 'results.json')
        best = engine.read_json(directory / 'metric_best.json')
        for objective in ('LPIPS', 'MS-SSIM'):
            row = best[objective]
            tag = row['snapshot']
            rows.append({'id': 'T3_reference_' + objective, 'kind': 'reference',
                         'run': str(directory), 'snapshot': tag,
                         'additional_iteration': row['iteration'],
                         'metrics': scores['ours_' + tag]})
    return rows


def configuration(preset, iterations=50000, off=False):
    runtime, text = engine.configuration(dict(T3, id=preset['id']), iterations, off)
    runtime.update(isotropic_weight=0. if off else preset['isotropic_weight'],
                   isotropic_scope='rendered', isotropic_eps=1e-8)
    return runtime, text


def initialize(preset, iterations, smoke=False, off=False):
    name = ('smoke_' if smoke else '') + preset['id']
    directory = RUNS / name / 'rubber_duck'
    runtime, text = configuration(preset, iterations, off)
    if directory.exists():
        if engine.read_json(directory / 'metric_config.json') != runtime or (directory / 'model_config.py').read_text(encoding='utf-8') != text:
            raise ValueError('Existing isotropic configuration changed')
        return directory
    directory.mkdir(parents=True)
    engine.write_json(directory / 'metric_config.json', runtime)
    engine.write_text_atomic(directory / 'model_config.py', text)
    if not off:
        for folder in ('point_cloud', 'deform'):
            shutil.copytree(engine.PARENTS['W256_D0'] / folder / 'iteration_best', directory / folder / 'iteration_best')
    engine.write_json(directory / 'reproduction.json', {
        'status': 'initialized', 'id': name, 'parent': runtime['parent_run'],
        'warm_start': not off, 'target_additional_iterations': iterations,
        'optimizer_initialization': 'fresh refinement; full optimizer/RNG restoration on resume',
        'resolution_width': 400 if off else 1600, 'seed': 0, 'official_test_count': 50,
        'original_train_count': 150, 'tuning': runtime, 'engineering_smoke_only': smoke,
        'source_freeze_sha256': engine.sha256(REPORTS / 'source_freeze.json'), 'commands': [],
    })
    return directory


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--smoke', action='store_true')
    choice.add_argument('--train', choices=[p['id'] for p in PRESETS])
    parser.add_argument('--hours', type=float, default=2)
    args = parser.parse_args()
    if args.hours <= 0:
        raise ValueError('Positive budget required')
    RUNS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    # Validate the immutable previous experiment before redirecting the coordinator.
    parent_repo = engine.prepare()
    engine.assert_frozen(parent_repo)
    repo = prepare()
    original_freeze = engine.freeze
    def freeze(path):
        value = original_freeze(path)
        value['isotropic_helpers'] = {str(p.relative_to(ROOT)): engine.sha256(p) for p in
                                      [HERE / name for name in ('model.py', 'runtime.py', 'prepare.py', 'run.py')]}
        value['metric_parent_source_freeze_sha256'] = engine.sha256(ROOT / 'experiments/metric_tuning/source_freeze.json')
        return value
    engine.RUNS, engine.REPORTS, engine.repo_global, engine.freeze = RUNS, REPORTS, repo, freeze
    engine.references = reference_rows
    engine.PRESETS = [{**T3, **p, 'id': ('smoke_' if args.smoke else '') + p['id']} for p in PRESETS]
    # Plan files make each full-length control independently runnable; no training yet.
    for preset in PRESETS:
        runtime, text = configuration(preset)
        engine.write_json(REPORTS / 'configs' / (preset['id'] + '.json'), runtime)
        engine.write_text_atomic(REPORTS / 'configs' / (preset['id'] + '.py'), text)
    frozen = freeze(repo)
    if (REPORTS / 'source_freeze.json').exists():
        assert engine.read_json(REPORTS / 'source_freeze.json') == frozen
    else:
        engine.write_json(REPORTS / 'source_freeze.json', frozen)
    engine.write_json(REPORTS / 'plan.json', {
        'presets': PRESETS, 'parent': str(engine.PARENTS['W256_D0']), 'scene': 'rubber_duck',
        'objective': 'T3 loss plus weight * mean(abs(abs(s)/mean(abs(s))-1))',
        'scope': 'actual rendered principal-axis magnitudes; canonical if Python covariance path is active',
        'evaluation': 'original 150 train / fixed official 50 test; 1600x900 PNG metrics',
        'control': 'I0 has the same parent, optimizer, schedule, LPIPS and resolution as I1-I3; weight is the only training change',
        'smoke_is_not_an_effectiveness_comparison': True,
    })
    deadline = datetime.now(timezone.utc) + timedelta(hours=args.hours)
    state = {'pid': os.getpid(), 'started_utc': engine.now(), 'deadline_utc': deadline.isoformat(),
             'status': 'validating' if args.smoke else 'training', 'mode': 'smoke' if args.smoke else args.train, 'completed': []}
    stop = threading.Event()
    def publish():
        engine.report()
        if args.smoke:
            notice = 'Engineering smoke validation only; these short-run scores do not establish reconstruction improvement.'
            path = REPORTS / 'results.md'
            engine.write_text_atomic(path, path.read_text(encoding='utf-8').replace('# PIDG multiobjective metric tuning', '# PIDG isotropy: engineering smoke validation\n\n' + notice, 1))
            path = REPORTS / 'monitor.html'
            engine.write_text_atomic(path, path.read_text(encoding='utf-8').replace('<h1>PIDG: four-metric optimization</h1>', '<h1>PIDG isotropy: engineering smoke validation</h1><p>' + notice + '</p>', 1))
    def monitor():
        while not stop.wait(5):
            try: publish()
            except (OSError, ValueError): pass
    with engine.lock():
        engine.write_json(RUNS / 'controller.json', state)
        watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
        try:
            if (RUNS / 'stop_queue.request').exists():
                raise RuntimeError('Pause request remains; acknowledge it before resuming')
            if args.smoke:
                off = initialize({'id': 'flags_off', 'isotropic_weight': 0.}, 20, True, True)
                assert engine.train(off, repo, deadline, 20, off=True, phase='flags_off') == 0
                state['completed'].append('flags_off:20')
                control = initialize(PRESETS[0], 100, True)
                assert engine.train(control, repo, deadline, 100, phase='smoke') == 0
                state['completed'].append('I0:100')
                regularized = initialize(PRESETS[2], 100, True)
                step = engine.read_json(regularized / 'checkpoints/resume.json')['iteration'] if (regularized / 'checkpoints/resume.json').exists() else 0
                if step < 20:
                    assert engine.train(regularized, repo, deadline, 100, limit=20, phase='smoke') == 75
                    assert engine.read_json(regularized / 'checkpoints/resume.json')['iteration'] == 20
                if step < 100:
                    assert engine.train(regularized, repo, deadline, 100, phase='resume_check') == 0
                assert engine.read_json(regularized / 'checkpoints/resume.json')['iteration'] == 100
                gradients = [__import__('json').loads(s) for s in (regularized / 'isotropic_gradient.jsonl').read_text().splitlines()]
                assert gradients and all(g['scale_gradient_l1'] > 0 for g in gradients)
                state['completed'].append('I2:20_to_100')
                engine.write_json(REPORTS / 'preflight.json', {
                    'status': 'passed', 'flags_off_steps': 20, 'T3_control_steps': 100,
                    'isotropic_steps': 100, 'full_resume_from': 20,
                    'finite_nonzero_regularization_gradients': True,
                    'original_physics_step_100_completed': True, 'completed_utc': engine.now(),
                    'quality_improvement_not_claimed': True,
                })
                state['status'] = 'preflight_passed'
            else:
                preset = next(p for p in PRESETS if p['id'] == args.train)
                directory = initialize(preset, 50000)
                code = engine.train(directory, repo, deadline - timedelta(minutes=20), phase='long')
                if code == 75:
                    state['status'] = 'saved_stopped'
                else:
                    engine.render_run(directory, repo, deadline)
                    state['status'] = 'completed'
            engine.assert_frozen(repo)
        except SystemExit as error:
            state['status'] = 'saved_stopped' if error.code == 75 else 'failed'
            if error.code != 75: raise
        except Exception as error:
            state.update(status='failed', error=str(error)); raise
        finally:
            state['finished_utc'] = engine.now()
            engine.write_json(RUNS / 'controller.json', state)
            stop.set(); watcher.join(timeout=6); publish()


if __name__ == '__main__':
    main()
