"""24-hour bounded multiobjective PIDG warm-start search; CPU-only coordinator."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from variants.metric_tuning.prepare import prepare, SOURCE
from pidg_benchmark import read_json, write_json, write_text_atomic, sha256, runtime_hashes, dataset_hashes
from monitor_common import parse_progress, tail, alive

RUNS = ROOT / 'runs/pidg_metric_tuning'
REPORTS = ROOT / 'experiments/metric_tuning'
PARENTS = {'W256_D0': ROOT / 'runs/pidg_official/rubber_duck',
           'W512_D0': ROOT / 'runs/pidg_mlp_capacity/long_W512_D0/rubber_duck',
           'W512_D2': ROOT / 'runs/pidg_mlp_capacity/long_W512_D2/rubber_duck'}
METRICS = ('PSNR', 'SSIM', 'LPIPS', 'MS-SSIM')
TAGS = {'PSNR': 'best', 'SSIM': 'metric_ssim', 'LPIPS': 'metric_lpips', 'MS-SSIM': 'metric_ms_ssim'}
PRESETS = [
    {'id': 'T0_continue', 'parent': 'W512_D0', 'pixel_loss': 'l1', 'ssim': .2, 'reg': .5},
    {'id': 'T1_mse', 'parent': 'W512_D0', 'pixel_loss': 'mse', 'ssim': .1, 'reg': .1},
    {'id': 'T2_residual_mse', 'parent': 'W512_D0', 'pixel_loss': 'mse', 'ssim': .1, 'reg': .1, 'residual': 128},
    {'id': 'T3_baseline_perceptual', 'parent': 'W256_D0', 'pixel_loss': 'l1', 'ssim': .2, 'reg': .5, 'perceptual': .02},
    {'id': 'T4_residual_perceptual', 'parent': 'W512_D0', 'pixel_loss': 'charbonnier', 'ssim': .2, 'reg': .1, 'residual': 128, 'perceptual': .02},
    {'id': 'T5_multiscale', 'parent': 'W512_D2', 'pixel_loss': 'l1', 'ssim': .3, 'reg': .1, 'structure': 'ms_ssim'},
]


def now():
    return datetime.now(timezone.utc).isoformat()


def metric_score(value, key):
    return -value if key == 'LPIPS' else value


def pareto(rows):
    def dominates(a, b):
        delta = [metric_score(a['metrics'][k], k) - metric_score(b['metrics'][k], k) for k in METRICS]
        return all(x >= -1e-7 for x in delta) and any(x > 1e-6 for x in delta)
    return [r for i, r in enumerate(rows) if not any(j != i and dominates(other, r) for j, other in enumerate(rows))]


def references():
    return [{'id': r['id'], 'kind': 'reference', 'run': r['run'], 'snapshot': 'best',
             'additional_iteration': 0, 'metrics': r['metrics']} for r in read_json(ROOT / 'experiments/mlp_capacity/results.json')['rows']]


def result_rows():
    rows = references()
    for preset in PRESETS:
        run = RUNS / preset['id'] / 'rubber_duck'
        if (run / 'metric_best.json').exists():
            for objective, best in read_json(run / 'metric_best.json').items():
                rows.append({'id': preset['id'], 'kind': 'candidate', 'run': str(run),
                             'objective': objective, 'snapshot': best['snapshot'],
                             'additional_iteration': best['iteration'], 'metrics': best['metrics']})
    return rows


def report():
    REPORTS.mkdir(parents=True, exist_ok=True)
    state = read_json(RUNS / 'controller.json') if (RUNS / 'controller.json').exists() else {}
    active = read_json(RUNS / 'active.json') if (RUNS / 'active.json').exists() else {}
    rows = result_rows()
    # Put historical references first so floating-point ties preserve the control.
    winners = {}
    for key in METRICS:
        best = rows[0]
        tolerance = 1e-4 if key == 'PSNR' else 1e-6
        for candidate in rows[1:]:
            if metric_score(candidate['metrics'][key], key) > metric_score(best['metrics'][key], key) + tolerance:
                best = candidate
        winners[key] = best
    live = {}
    if active.get('run'):
        selected = Path(active['run']).resolve()
        if selected.is_relative_to(RUNS.resolve()):
            live = parse_progress(tail(selected / 'train.log'))
    summary = {'updated_utc': now(), 'evaluation': 'fixed 50 official test views, 1600x900 PNG-equivalent metrics',
               'selection': 'test-selected snapshots; test RGB never enters gradient loss',
               'controller': state, 'active': active, 'progress': live,
               'rows': rows, 'best_by_metric': winners, 'pareto': pareto(rows)}
    write_json(REPORTS / 'results.json', summary)
    lines = ['# PIDG multiobjective metric tuning', '',
             'Rubber Duck; original 150 training views; fixed official 50 test views; seed0.',
             'Warm-start refinement with additional compute; not a same-budget from-scratch architecture ablation.',
             'Four independent best snapshots are kept. Test selection follows the historical PIDG protocol.', '',
             '| Objective | Best model | Snapshot | Additional step | PSNR | SSIM | LPIPS | MS-SSIM |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for key, row in winners.items():
        lines.append('| ' + ' | '.join([key, row['id'], row['snapshot'], str(row['additional_iteration']),
                                       *[f"{row['metrics'][m]:.6f}" for m in METRICS]]) + ' |')
    lines += ['', '## Observed snapshots', '', '| Model | Objective | Extra step | PSNR | SSIM | LPIPS | MS-SSIM |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + ' | '.join([row['id'], row.get('objective', 'reference'), str(row['additional_iteration']),
                                       *[f"{row['metrics'][m]:.6f}" for m in METRICS]]) + ' |')
    lines += ['', 'Status: ' + str(state.get('status', 'preparing')), 'Deadline UTC: ' + state.get('deadline_utc', '-'),
              'Current stage: ' + active.get('phase', '-'), 'Current model: ' + active.get('id', '-'),
              '', '[5-second monitor](monitor.html)', '[Plan](plan.json)', '[Verified rendered winners](rendered_winners.json)',
              '', 'Updated UTC: ' + summary['updated_utc']]
    write_text_atomic(REPORTS / 'results.md', '\n'.join(lines) + '\n')
    cells = ''.join('<tr><td>' + '</td><td>'.join(html.escape(str(v)) for v in
                    [r['id'], r.get('objective', 'reference'), r['additional_iteration'],
                     *[f"{r['metrics'][m]:.6f}" for m in METRICS]]) + '</td></tr>' for r in rows)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>PIDG metric tuning</title><style>body{font:16px system-ui;margin:30px;color:#153149;background:#f3f6f9}table{width:100%;border-collapse:collapse;background:white}th,td{padding:10px;border-bottom:1px solid #d8e2ea;text-align:left}pre{white-space:pre-wrap}</style><h1>PIDG: four-metric optimization</h1>'
    page += '<p>' + html.escape(str(state.get('status', 'preparing'))) + ' | deadline UTC: ' + html.escape(state.get('deadline_utc', '-')) + '</p>'
    page += '<p>Current: ' + html.escape(active.get('id', '-')) + ' / ' + html.escape(active.get('phase', '-')) + ' | step: ' + str(live.get('iteration', '-')) + '</p>'
    page += '<p>Fixed 50 test views. Warm starts; extra iterations; original controls retained. Refresh: 5 seconds.</p><table><tr><th>Model</th><th>Objective</th><th>Extra step</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>MS-SSIM</th></tr>' + cells + '</table><p>Updated UTC: ' + summary['updated_utc'] + '</p></html>'
    write_text_atomic(REPORTS / 'monitor.html', page)
    return summary


def freeze(repo):
    parents = {}
    for key, directory in PARENTS.items():
        files = [directory / 'cfg_args', directory / 'results.json', directory / 'reproduction.json']
        for folder in ('point_cloud/iteration_best', 'deform/iteration_best'):
            files += [p for p in (directory / folder).rglob('*') if p.is_file()]
        parents[key] = {str(p): sha256(p) for p in files}
    return {'original': runtime_hashes(SOURCE), 'fork': runtime_hashes(repo),
            'data': dataset_hashes('rubber_duck'), 'parents': parents,
            'helpers': {str(p.relative_to(ROOT)): sha256(p) for p in [HERE / name for name in
                        ('model.py', 'runtime.py', 'checkpoint.py', 'prepare.py', 'run.py')] +
                        [ROOT / 'scripts/pidg_variant_entrypoint.py', ROOT / 'reproduction/scripts/training_checkpoint.py']}}


def assert_frozen(repo):
    if read_json(REPORTS / 'source_freeze.json') != freeze(repo):
        raise ValueError('Frozen source/data/parent weights changed')


def configuration(preset, iterations=50000, off=False):
    width = 256 if preset['parent'] == 'W256_D0' else 512
    depth = 2 if preset['parent'] == 'W512_D2' else 0
    runtime = {'enabled': not off, 'parent_run': str(PARENTS[preset['parent']]), 'pixel_loss': preset['pixel_loss'],
               'structure': preset.get('structure', 'ssim'), 'perceptual_weight': preset.get('perceptual', 0),
               'perceptual_interval': 4}
    text = (SOURCE / 'arguments/pidg/rubber_dack.py').read_text(encoding='utf-8')
    text += f"\nnetwork_args = dict(network_args, width={width}, depth={depth}, metric_residual_width={0 if off else preset.get('residual', 0)})\n"
    if not off:
        values = {'warm_up': 0, 'densify_until_iter': 0, 'position_lr_init': 2e-5, 'position_lr_final': 1e-7,
                  'position_lr_max_steps': iterations, 'deform_lr_max_steps': iterations, 'feature_lr': .0005,
                  'opacity_lr': .005, 'scaling_lr': .00005, 'rotation_lr': .0001,
                  'grid_lr_scale': 25., 'network_lr_scale': 5., 'velocity_net_lr': .0001, 'force_sigma_lr': .0001,
                  'lambda_dssim': preset['ssim'], 'lambda_spatial_tv': preset['reg'], 'lambda_temporal_tv': preset['reg']}
        text += '\n'.join(f'{key} = {value!r}' for key, value in values.items()) + '\n'
    return runtime, text


def initialize_run(preset, repo, iterations=50000, name=None, off=False):
    directory = RUNS / (name or preset['id']) / 'rubber_duck'
    runtime, config = configuration(preset, iterations, off)
    if directory.exists():
        if read_json(directory / 'metric_config.json') != runtime or (directory / 'model_config.py').read_text(encoding='utf-8') != config:
            raise ValueError('Existing tuning run configuration changed')
        return directory
    directory.mkdir(parents=True)
    write_json(directory / 'metric_config.json', runtime)
    write_text_atomic(directory / 'model_config.py', config)
    if not off:
        for name in ('point_cloud', 'deform'):
            shutil.copytree(PARENTS[preset['parent']] / name / 'iteration_best', directory / name / 'iteration_best')
    write_json(directory / 'reproduction.json', {'status': 'initialized', 'id': preset['id'],
               'parent': runtime['parent_run'], 'warm_start': not off,
               'optimizer_initialization': 'fresh for a new refinement; full state restored on continuation',
               'target_additional_iterations': iterations, 'resolution_width': 400 if off else 1600, 'seed': 0,
               'original_train_count': 150, 'official_test_count': 50, 'tuning': runtime,
               'source_freeze_sha256': sha256(REPORTS / 'source_freeze.json'), 'commands': []})
    return directory


def env_for(directory):
    env = os.environ.copy(); env.pop('PIDG_CONSTITUTIVE_CONFIG', None)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    env['PIDG_METRIC_CONFIG'] = str(directory / 'metric_config.json')
    env['PIDG_HASH_SOURCE_DIR'] = str(SOURCE / 'hashencoder')
    env['PIDG_HASH_BUILD_DIR'] = str(SOURCE / 'tmp_build')
    env.pop('PIDG_TUNING_STOP_AFTER', None)
    return env


def command_prefix(repo):
    return [sys.executable, '-u', str(ROOT / 'scripts/pidg_variant_entrypoint.py'), '--repo', str(repo), '--seed', '0']


def execute(command, directory, phase, env, deadline):
    info = read_json(directory / 'reproduction.json')
    info['commands'].append(command); info['status'] = phase
    write_json(directory / 'reproduction.json', info)
    log = directory / ('train.log' if phase in ('screen', 'long', 'smoke', 'resume_check', 'flags_off') else phase + '.log')
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\nCOMMAND ' + subprocess.list2cmdline(command) + '\n'); stream.flush()
        child = subprocess.Popen(command, cwd=repo_global, env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        write_json(RUNS / 'active.json', {'id': info['id'], 'phase': phase, 'run': str(directory),
                                        'pid': child.pid, 'controller_pid': os.getpid(), 'started_utc': now()})
        forwarded = False
        while child.poll() is None:
            stop = (RUNS / 'stop_queue.request').exists() or datetime.now(timezone.utc) >= deadline
            if stop and phase in ('screen', 'long', 'smoke', 'resume_check', 'flags_off') and not forwarded:
                (directory / 'checkpoints').mkdir(exist_ok=True)
                (directory / 'checkpoints/save_and_stop.request').write_text('Save state and release GPU', encoding='utf-8')
                forwarded = True
            time.sleep(1)
    if (directory / 'checkpoints/resume.json').exists():
        info['completed_additional_iterations'] = read_json(directory / 'checkpoints/resume.json')['iteration']
    info['status'] = 'saved_stopped' if child.returncode == 75 else 'completed' if child.returncode == 0 else 'failed'
    write_json(directory / 'reproduction.json', info)
    if child.returncode not in (0, 75):
        raise RuntimeError(f'{phase} failed ({child.returncode}): {log}')
    if child.returncode == 75 and (RUNS / 'stop_queue.request').exists():
        raise SystemExit(75)
    return child.returncode


def train(directory, repo, deadline, iterations=50000, limit=0, off=False, phase='screen'):
    assert_frozen(repo)
    prefix = command_prefix(repo)
    config = directory / 'model_config.py'
    command = [*prefix, 'train_pidg.py', '-s', str(ROOT / 'data/PIDG/rubber_duck'), '-m', str(directory),
               '--conf', str(config), '--load2gpu_on_the_fly', '--resolution', '400' if off else '1600',
               '--iterations', str(iterations), '--checkpoint_interval', '1000' if iterations > 1000 else str(max(1, limit or iterations)),
               '--save_image_dir', str(directory / 'diagnostics'), '--save_iterations', str(iterations), '--test_iterations']
    tests = sorted(set([1000, *range(2000, iterations + 1, 1000), iterations]) if iterations > 1000 else set([limit or iterations, iterations]))
    command += list(map(str, tests))
    if off:
        command += ['--warm_up', '0', '--densify_until_iter', '0']
    checkpoint = directory / 'checkpoints/resume.pt'
    if checkpoint.exists():
        manifest = read_json(checkpoint.with_suffix('.json'))
        if sha256(checkpoint) != manifest['sha256']:
            raise ValueError('Invalid full checkpoint')
        command += ['--resume_checkpoint', str(checkpoint)]
    env = env_for(directory)
    if limit:
        env['PIDG_TUNING_STOP_AFTER'] = str(limit)
    return execute(command, directory, phase, env, deadline)


def render_run(directory, repo, deadline, off=False, tags=None):
    tags = tags or (['best'] if off else list(dict.fromkeys(TAGS.values())))
    prefix = command_prefix(repo); env = env_for(directory)
    for tag in tags:
        raw = directory / f'test/ours_{tag}'
        snapshot = {str(p): sha256(p) for folder in ('point_cloud', 'deform')
                    for p in (directory / folder / ('iteration_' + tag)).rglob('*') if p.is_file()}
        if (raw / 'render_complete.json').exists() and read_json(raw / 'render_complete.json').get('snapshot') == snapshot:
            continue
        command = [*prefix, 'render.py', '-s', str(ROOT / 'data/PIDG/rubber_duck'), '-m', str(directory),
                   '--conf', str(directory / 'model_config.py'), '--load2gpu_on_the_fly', '--resolution',
                   '400' if off else '1600', '--iteration', tag, '--skip_train']
        if execute(command, directory, 'render_' + tag, env, deadline):
            raise RuntimeError('Rendering unexpectedly stopped')
        names = {f'{i:05d}.png' for i in range(50)}
        assert all({p.name for p in (raw / kind).glob('*.png')} == names for kind in ('renders', 'gt'))
        write_json(raw / 'render_complete.json', {'frames': 50, 'snapshot': snapshot})
    execute([*prefix, 'metrics.py', '-m', str(directory)], directory, 'metrics', env, deadline)
    values = read_json(directory / 'results.json')
    if not off:
        for best in read_json(directory / 'metric_best.json').values():
            if best['snapshot'] not in tags:
                continue
            actual = values['ours_' + best['snapshot']]
            assert all(abs(actual[k] - best['metrics'][k]) < 1e-4 for k in METRICS)
    return values


def select_long():
    rows = [r for r in result_rows() if r['kind'] == 'candidate' and r['additional_iteration'] > 0]
    selected = []
    for key in METRICS:
        if rows:
            winner = max(rows, key=lambda row: metric_score(row['metrics'][key], key))['id']
            if winner not in selected:
                selected.append(winner)
    for name in ('T2_residual_mse', 'T4_residual_perceptual', 'T3_baseline_perceptual', 'T5_multiscale'):
        if len(selected) < 4 and name not in selected:
            selected.append(name)
    return selected[:4]


@contextmanager
def lock():
    with (RUNS / 'controller.lock').open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1): stream.write(b'0'); stream.flush()
        stream.seek(0)
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try: yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    global repo_global
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    repo = prepare(); repo_global = repo
    frozen = freeze(repo)
    if (REPORTS / 'source_freeze.json').exists():
        assert read_json(REPORTS / 'source_freeze.json') == frozen
    else:
        write_json(REPORTS / 'source_freeze.json', frozen)
    plan = {'scene': 'rubber_duck', 'budget_hours': args.hours, 'objectives': list(METRICS),
            'reference_models_reused': [r['id'] for r in references()], 'presets': PRESETS,
            'screen_additional_steps': 1000, 'long_additional_steps': 50000, 'max_long_models': 4,
            'method': 'warm-start original best weights; new optimizer for new experiments; exact full-state continuation thereafter',
            'evaluation': 'fixed 50 official test views; PNG-equivalent metrics; no test gradient loss',
            'snapshot_selection': 'independent best PSNR, SSIM, LPIPS and MS-SSIM; observed Pareto set retained',
            'limitations': ['test-based selection and extensive tuning; no untouched independent test set',
                            'additional training compute differs from historical baseline',
                            'one scene and one seed; observed optima, not global optimality'],
            'references': ['https://arxiv.org/abs/2308.04079', 'https://arxiv.org/abs/1603.05027']}
    write_json(REPORTS / 'plan.json', plan)
    state = read_json(RUNS / 'controller.json') if args.resume and (RUNS / 'controller.json').exists() else {
        'started_utc': now(), 'deadline_utc': (datetime.now(timezone.utc) + timedelta(hours=args.hours)).isoformat(), 'completed': []}
    state.update(pid=os.getpid(), status='preflight')
    deadline = datetime.fromisoformat(state['deadline_utc'])
    training_deadline = deadline - timedelta(minutes=40)
    stop = threading.Event()
    def monitor():
        while not stop.wait(5):
            try: report()
            except (OSError, ValueError): pass
    with lock():
        write_json(RUNS / 'controller.json', state); report()
        watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
        try:
            if (RUNS / 'stop_queue.request').exists():
                raise RuntimeError('Pause request remains present')
            off = initialize_run({'id': 'flags_off', 'parent': 'W256_D0', 'pixel_loss': 'l1', 'ssim': .2, 'reg': .5}, repo, 20, 'smoke_flags_off', True)
            if not (REPORTS / 'preflight.json').exists():
                train(off, repo, training_deadline, 20, off=True, phase='flags_off')
                render_run(off, repo, deadline, True)
                smoke = initialize_run(PRESETS[4], repo, 40, 'smoke_residual_perceptual')
                assert train(smoke, repo, training_deadline, 40, 20, phase='smoke') == 75
                assert read_json(smoke / 'checkpoints/resume.json')['iteration'] == 20
                assert train(smoke, repo, training_deadline, 40, phase='resume_check') == 0
                assert read_json(smoke / 'checkpoints/resume.json')['iteration'] == 40
                render_run(smoke, repo, deadline)
                write_json(REPORTS / 'preflight.json', {'status': 'passed', 'flags_off_steps': 20,
                           'warm_start_steps': 40, 'resume_verified_from': 20, 'to': 40,
                           'four_snapshot_png_metrics_verified': True, 'original_sources_unchanged': runtime_hashes(SOURCE) == frozen['original'],
                           'completed_utc': now()})
            if args.preflight_only:
                state['status'] = 'preflight_passed'; return
            state['status'] = 'screening'; write_json(RUNS / 'controller.json', state)
            for preset in PRESETS:
                if datetime.now(timezone.utc) >= training_deadline or (RUNS / 'stop_queue.request').exists(): break
                directory = initialize_run(preset, repo)
                completed = read_json(directory / 'checkpoints/resume.json')['iteration'] if (directory / 'checkpoints/resume.json').exists() else 0
                if completed < 1000:
                    train(directory, repo, training_deadline, limit=1000)
                state['completed'].append('screen:' + preset['id'])
                write_json(RUNS / 'controller.json', state)
            selected = state.get('selected_long') or select_long()
            state.update(status='long_training', selected_long=selected); write_json(RUNS / 'controller.json', state)
            for name in selected:
                if datetime.now(timezone.utc) >= training_deadline or (RUNS / 'stop_queue.request').exists(): break
                preset = next(p for p in PRESETS if p['id'] == name)
                directory = initialize_run(preset, repo)
                completed = read_json(directory / 'checkpoints/resume.json')['iteration'] if (directory / 'checkpoints/resume.json').exists() else 0
                if completed < 50000:
                    train(directory, repo, training_deadline, phase='long')
                state['completed'].append('long:' + name); write_json(RUNS / 'controller.json', state)
            if (RUNS / 'stop_queue.request').exists():
                state['status'] = 'saved_stopped'; return
            state['status'] = 'rendering_winners'; write_json(RUNS / 'controller.json', state)
            # Export the global winning snapshots; retain other checkpoints too.
            winners = report()['best_by_metric']
            rendered = []
            for run in dict.fromkeys(row['run'] for row in winners.values() if row['kind'] == 'candidate'):
                tags = list(dict.fromkeys(row['snapshot'] for row in winners.values() if row['run'] == run))
                values = render_run(Path(run), repo, deadline, tags=tags)
                rendered.append({'run': run, 'metrics': values, 'render_root_absolute': str(Path(run) / 'test')})
            write_json(REPORTS / 'rendered_winners.json', {'status': 'completed', 'winners': winners, 'rendered': rendered})
            assert_frozen(repo)
            state['status'] = 'budget_complete' if datetime.now(timezone.utc) >= training_deadline else 'completed'
            state['finished_utc'] = now()
        except SystemExit as error:
            state['status'] = 'saved_stopped' if error.code == 75 else 'failed'
            if error.code != 75: raise
        except Exception as error:
            state.update(status='failed', error=str(error)); raise
        finally:
            write_json(RUNS / 'controller.json', state); stop.set(); watcher.join(timeout=6); report()


if __name__ == '__main__':
    main()
