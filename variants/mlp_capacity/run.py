"""Baseline PIDG deformation-MLP capacity ablation; no constitutive extensions.

Original runtime files are never edited. Per-run copies of the scene config only
override DeformNetwork width/depth. The existing completed baseline is reused.
"""
import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from pidg_benchmark import completed_metrics, dataset_hashes, read_json, runtime_hashes, sha256, write_json, write_text_atomic
from monitor_common import alive, parse_progress, tail
import run_constitutive_experiment as execution

REPO = ROOT / 'third_party/PIDG'
RUNS = ROOT / 'runs/pidg_mlp_capacity'
REPORTS = ROOT / 'experiments/mlp_capacity'
BASE_CONFIG = REPO / 'arguments/pidg/rubber_dack.py'
BASELINE = ROOT / 'runs/pidg_official/rubber_duck'
PRESETS = [('W256_D0', 256, 0), ('W512_D0', 512, 0), ('W256_D2', 256, 2), ('W512_D2', 512, 2)]
# Existing executor uses this root exclusively for its live status/stop requests.
# This process does not launch the old constitutive queue.
execution.RUNS = RUNS


def now():
    return datetime.now(timezone.utc).isoformat()


def variants():
    return [{'id': name, 'width': width, 'depth': depth,
             'mlp_parameters': 240 * width + 14 + depth * (width * width + width)}
            for name, width, depth in PRESETS]


def resolved_config(item):
    text = BASE_CONFIG.read_text(encoding='utf-8')
    return text + '\n# Capacity ablation: change only the original deformation MLP.\n' + (
        f"network_args = dict(network_args, width={item['width']}, depth={item['depth']})\n")


def initialize():
    REPORTS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (HERE / 'configs').mkdir(exist_ok=True)
    for item in variants():
        path = HERE / 'configs' / (item['id'] + '.json')
        if path.exists():
            if read_json(path) != item:
                raise ValueError('Existing preset changed: ' + str(path))
        else:
            write_json(path, item)
    plan = {'scene': 'rubber_duck', 'backbone': 'original released PIDG compatibility baseline',
            'physics_extensions': 'disabled; original trainer, no pidg_ext training hooks',
            'changed_module': 'scene_PIDG.network.DeformNetwork only',
            'fixed_modules': ['hash grids', 'velocity/stress network', 'Gaussian model', 'renderer'],
            'long_protocol': {'iterations': 50000, 'resolution_width': 1600, 'seed': 0},
            'smoke_protocol': {'iterations': 300, 'resolution_width': 400, 'seed': 0,
                               'warm_up': 0, 'densify_until_iter': 0},
            'variants': variants(), 'baseline_reuse': str(BASELINE),
            'selection': 'original best-test-PSNR protocol; also report final-50k logged PSNR',
            'limitations': ['single scene and seed; exploratory comparison',
                            'historical baseline did not record runtime source hashes',
                            'different widths/depths have different initialization draws and runtime costs']}
    if (REPORTS / 'plan.json').exists() and read_json(REPORTS / 'plan.json') != plan:
        raise ValueError('Existing ablation plan changed')
    write_json(REPORTS / 'plan.json', plan)


def report():
    rows = []
    for item in variants():
        if item['id'] == 'W256_D0':
            directory = BASELINE
            info, metrics = completed_metrics(directory)
            state = 'completed (reused)'
        else:
            directory = RUNS / ('long_' + item['id']) / 'rubber_duck'
            info = read_json(directory / 'reproduction.json') if (directory / 'reproduction.json').exists() else {}
            state = info.get('status', 'pending')
            metrics = read_json(directory / 'results.json').get('ours_best') if state == 'completed' else None
        progress = parse_progress(tail(directory / 'train.log'))
        final = next((x['value'] for x in reversed(progress['test_psnr']) if x['iteration'] == 50000), None)
        rows.append({**item, 'run': str(directory), 'status': state,
                     'iteration': info.get('iterations', 50000) if state.startswith('completed') else progress['iteration'],
                     'metrics': metrics, 'final_50k_logged_psnr': final})
    active = read_json(RUNS / 'active.json') if (RUNS / 'active.json').exists() else {}
    controller = read_json(RUNS / 'controller.json') if (RUNS / 'controller.json').exists() else {}
    snapshot = {'updated_utc': now(), 'refresh_seconds': 5, 'rows': rows, 'controller': controller,
                'controller_alive': alive(controller.get('pid')), 'active': active,
                'child_alive': alive(active.get('child_pid'))}
    base_psnr = rows[0]['metrics']['PSNR']
    header = ['# Baseline PIDG: deformation MLP capacity ablation', '',
              'Rubber Duck; 50k; width1600; seed0. F/NH/material/force extensions are OFF.',
              'W256_D0 reuses the completed original baseline. Smoke results are not long-run results.', '',
              '| Variant | MLP params | Status | Iteration | Best PSNR | Delta | Final-50k logged PSNR |',
              '|---|---:|---|---:|---:|---:|---:|']
    def fmt(value):
        return '-' if value is None else f'{value:.6f}'
    for row in rows:
        value = row['metrics']['PSNR'] if row['metrics'] else None
        header.append(f"| {row['id']} | {row['mlp_parameters']:,} | {row['status']} | {row['iteration'] or '-'} | {fmt(value)} | {fmt(value-base_psnr) if value is not None else '-'} | {fmt(row['final_50k_logged_psnr'])} |")
    header += ['', 'No conclusion about a PSNR gain is made before the corresponding long run finishes.',
               'Single-seed differences are descriptive, not statistically established. Best snapshots use test-based selection.',
               '', '[Five-second monitor](monitor.html)', '', 'Runs: `runs/pidg_mlp_capacity/`. Configurations: `variants/mlp_capacity/configs/`.',
               '', 'Updated UTC: ' + snapshot['updated_utc']]
    write_json(REPORTS / 'results.json', snapshot)
    write_text_atomic(REPORTS / 'results.md', '\n'.join(header) + '\n')
    cells = ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in
                     [r['id'], f"{r['mlp_parameters']:,}", r['status'], r['iteration'] or '-',
                      fmt(r['metrics']['PSNR']) if r['metrics'] else '-', fmt(r['final_50k_logged_psnr'])]) + '</tr>' for r in rows)
    current_log = ''
    if active.get('run'):
        selected = Path(active['run']).resolve()
        if selected.is_relative_to(RUNS.resolve()):
            current_log = '\n'.join(tail(selected / (active.get('step', 'train') + '.log'), 16000).replace('\r', '\n').splitlines()[-14:])
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>PIDG MLP capacity monitor</title><style>body{font:16px system-ui;margin:36px;background:#f3f6f9;color:#153149}table{border-collapse:collapse;background:white;width:100%}th,td{padding:12px;border-bottom:1px solid #dbe4ec;text-align:left}pre{background:#153149;color:#e5f3f5;padding:18px;white-space:pre-wrap}small{color:#637487}</style>
<h1>Baseline PIDG: MLP capacity</h1><p>Rubber Duck / 50k / seed0 / physics extensions OFF. Refresh: 5 seconds.</p>'''
    page += '<p>Controller: ' + html.escape(controller.get('status', 'not started')) + ' | Current stage: ' + html.escape(active.get('step', '-')) + '</p>'
    page += '<table><tr><th>Variant</th><th>MLP parameters</th><th>Status</th><th>Iteration</th><th>Best PSNR</th><th>Final logged PSNR</th></tr>' + cells + '</table>'
    page += '<h2>Current log</h2><pre>' + html.escape(current_log) + '</pre><small>Updated UTC: ' + snapshot['updated_utc'] + '</small></html>'
    write_text_atomic(REPORTS / 'monitor.html', page)
    return snapshot


def run_one(item, smoke=False, resume=False):
    directory = RUNS / (('smoke_' if smoke else 'long_') + item['id']) / 'rubber_duck'
    spec = {'iterations': 300 if smoke else 50000, 'resolution_width': 400 if smoke else 1600,
            'seed': 0, 'smoke_only': smoke, 'profile': 'engineering_smoke' if smoke else 'official',
            'scene': 'rubber_duck', 'config': {'deformation_gradient': False, 'constitutive': False,
                'learn_material_parameters': False, 'multi_material': False, 'structured_force': False,
                'network_args': {'width': item['width'], 'depth': item['depth'], 'directional': True}},
            'source_hashes': runtime_hashes(REPO), 'dataset_hashes': dataset_hashes('rubber_duck'),
            'base_config_sha256': sha256(BASE_CONFIG), 'runner_sha256': sha256(__file__),
            'shared_checkpoint_helper_sha256': sha256(ROOT / 'reproduction/scripts/training_checkpoint.py'),
            'entrypoint_sha256': sha256(ROOT / 'scripts/pidg_variant_entrypoint.py')}
    config_text = resolved_config(item)
    checkpoint_path = None
    if directory.exists():
        if not resume:
            raise FileExistsError('Existing run preserved; use --resume: ' + str(directory))
        info = read_json(directory / 'reproduction.json')
        if any(info.get(k) != v for k, v in spec.items()):
            raise ValueError('Resume source/data/config/protocol mismatch')
        if (directory / 'model_config.py').read_text(encoding='utf-8') != config_text:
            raise ValueError('Saved architecture config changed')
        if info['status'] == 'completed':
            completed_metrics(directory)
            return
        if 'train' not in info['completed_steps']:
            checkpoint = execution.verified_checkpoint(directory)
            checkpoint_path = directory / 'checkpoints/resume.pt'
            info.setdefault('resume_history', []).append({'iteration': checkpoint['iteration'], 'utc': now()})
    else:
        directory.mkdir(parents=True)
        write_text_atomic(directory / 'model_config.py', config_text)
        info = {**spec, 'experiment': 'PIDG_MLP_CAPACITY', 'configuration_id': item['id'], 'mlp_parameters': item['mlp_parameters'],
                'completed_steps': [], 'commands': [], 'started_utc': now(), 'resolved_config_sha256': sha256(directory / 'model_config.py')}
    env = os.environ.copy()
    env.pop('PIDG_CONSTITUTIVE_CONFIG', None)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    env['PIDG_HASH_SOURCE_DIR'] = str(REPO / 'hashencoder')
    env['PIDG_HASH_BUILD_DIR'] = str(REPO / 'tmp_build')
    prefix = [sys.executable, '-u', str(ROOT / 'scripts/pidg_variant_entrypoint.py'), '--repo', str(REPO), '--seed', '0']
    shared = ['-s', str(ROOT / 'data/PIDG/rubber_duck'), '-m', str(directory), '--conf', str(directory / 'model_config.py'),
              '--load2gpu_on_the_fly', '--resolution', str(spec['resolution_width'])]
    train = [*prefix, 'train_pidg.py', *shared, '--iterations', str(spec['iterations']), '--checkpoint_interval',
             '300' if smoke else '1000', '--save_image_dir', str(directory / 'diagnostics')]
    if smoke:
        train += ['--warm_up', '0', '--densify_until_iter', '0', '--test_iterations', '300', '--save_iterations', '300']
    if checkpoint_path is not None:
        train += ['--resume_checkpoint', str(checkpoint_path)]
    commands = [('train', train), ('render', [*prefix, 'render.py', *shared, '--iteration', 'best', '--skip_train']),
                ('metrics', [*prefix, 'metrics.py', '-m', str(directory)])]
    try:
        for step, command in commands:
            if step in info['completed_steps']:
                continue
            if (RUNS / 'stop_queue.request').exists():
                raise SystemExit(75)
            info['status'] = {'train': 'training', 'render': 'rendering', 'metrics': 'evaluating'}[step]
            write_json(directory / 'reproduction.json', info)
            code = execution.execute(command, directory, step, REPO, env, info)
            if code == 75:
                raise SystemExit(75)
            if code:
                raise RuntimeError(f'{step} failed ({code}); see {directory / step}.log')
            info['completed_steps'].append(step)
            write_json(directory / 'reproduction.json', info)
        assert runtime_hashes(REPO) == spec['source_hashes']
        assert dataset_hashes('rubber_duck') == spec['dataset_hashes']
        assert sha256(directory / 'model_config.py') == info['resolved_config_sha256']
        assert execution.verified_checkpoint(directory)['iteration'] == spec['iterations']
        assert 'Training complete.' in tail(directory / 'train.log')
        metrics = read_json(directory / 'results.json')
        info.update(status='completed', metrics=metrics, finished_utc=now())
        write_json(directory / 'reproduction.json', info)
        completed_metrics(directory)
    except SystemExit:
        info['status'] = 'saved_stopped'
        write_json(directory / 'reproduction.json', info)
        raise
    except Exception as error:
        info.update(status='failed', error=str(error))
        write_json(directory / 'reproduction.json', info)
        raise


def work(mode, resume=False):
    initialize()
    if (RUNS / 'stop_queue.request').exists():
        raise RuntimeError('Saved pause request present; acknowledge it before resuming')
    # Exclusive lock is scoped to this new queue, not to old constitutive runs.
    import run_constitutive_matrix as locks
    locks.RUNS = RUNS
    controller = {'pid': os.getpid(), 'status': 'running', 'mode': mode, 'started_utc': now(),
                  'completed_jobs': [], 'runner_sha256': sha256(__file__)}
    stop = threading.Event()
    def refresh():
        while not stop.wait(5):
            try:
                report()
            except Exception as error:
                print('Monitor:', error, flush=True)
    with locks.controller_lock():
        write_json(RUNS / 'controller.json', controller)
        thread = threading.Thread(target=refresh, daemon=True)
        thread.start()
        try:
            info, _ = completed_metrics(BASELINE)
            if (info['iterations'], info['resolution_width'], info['seed']) != (50000, 1600, 0):
                raise ValueError('Reused baseline protocol mismatch')
            if mode in ('all', 'smoke'):
                controller['current_job'] = 'architecture_and_flags_off_validation'
                write_json(RUNS / 'controller.json', controller)
                with (RUNS / 'validation.log').open('a', encoding='utf-8') as stream:
                    subprocess.run([sys.executable, '-u', str(HERE / 'validate.py')], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                for item in variants():
                    controller['current_job'] = 'smoke_' + item['id']
                    write_json(RUNS / 'controller.json', controller)
                    run_one(item, smoke=True, resume=resume)
                    controller['completed_jobs'].append(controller['current_job'])
            if mode in ('all', 'long'):
                for item in variants():
                    check = RUNS / ('smoke_' + item['id']) / 'rubber_duck'
                    completed_metrics(check)
                for item in variants()[1:]:
                    if shutil.disk_usage(ROOT).free < 15 * 1024 ** 3:
                        raise RuntimeError('Less than 15 GiB free')
                    controller['current_job'] = 'long_' + item['id']
                    write_json(RUNS / 'controller.json', controller)
                    run_one(item, smoke=False, resume=resume)
                    controller['completed_jobs'].append(controller['current_job'])
                    write_json(RUNS / 'controller.json', controller)
            controller['status'] = 'completed'
        except SystemExit as error:
            controller['status'] = 'saved_stopped' if error.code == 75 else 'failed'
            if error.code != 75:
                raise
        except Exception as error:
            controller.update(status='failed', error=str(error))
            raise
        finally:
            stop.set();thread.join(timeout=6)
            controller['updated_utc'] = now()
            write_json(RUNS / 'controller.json', controller)
            report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all', 'smoke', 'long', 'report'], default='all')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.mode == 'report':
        initialize();report()
    else:
        work(args.mode, args.resume)
