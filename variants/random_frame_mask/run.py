"""CPU-only dependent queue; original PIDG baseline with random missing frames."""
import argparse
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import DATA, MASK_SEED, RATIOS, REPORTS, SOURCE, fixture_hashes, link_directory, prepare, save_frozen_json
sys.path.insert(0, str(ROOT / 'scripts'))
from pidg_benchmark import completed_metrics, read_json, runtime_hashes, sha256, write_json, write_text_atomic
from monitor_common import alive, parse_progress, tail
import run_constitutive_experiment as execution

RUNS = ROOT / 'runs/pidg_random_frame_mask'
MLP_RUNS = ROOT / 'runs/pidg_mlp_capacity'
BASELINE = ROOT / 'runs/pidg_official/rubber_duck'
REPO = ROOT / 'third_party/PIDG'
BASE_CONFIG = REPO / 'arguments/pidg/rubber_dack.py'
execution.RUNS = RUNS


def now():
    return datetime.now(timezone.utc).isoformat()


def frame_outputs_complete(directory, count):
    expected = {f'{i:05d}.png' for i in range(count)}
    return all({p.name for p in (Path(directory) / 'test/ours_best' / kind).glob('*.png')} == expected
               for kind in ('renders', 'gt'))


def dependency_ready(mlp_root=MLP_RUNS, baseline=BASELINE):
    """Never use progress counters alone; require finalized metrics and image sets."""
    mlp_root, baseline = Path(mlp_root), Path(baseline)
    if not (mlp_root / 'controller.json').exists():
        return False, 'MLP controller has not been created'
    controller = read_json(mlp_root / 'controller.json')
    if controller.get('status') != 'completed':
        return False, 'MLP queue state: ' + str(controller.get('status'))
    if (mlp_root / 'stop_queue.request').exists():
        return False, 'MLP pause request remains present'
    if alive(controller.get('pid')):
        return False, 'Waiting for the MLP controller to exit'
    active = read_json(mlp_root / 'active.json') if (mlp_root / 'active.json').exists() else {}
    if alive(active.get('child_pid')):
        return False, 'Waiting for the last MLP training/render/metrics process to exit'
    for directory in [baseline] + [mlp_root / ('long_' + name) / 'rubber_duck'
                                   for name in ('W512_D0', 'W256_D2', 'W512_D2')]:
        try:
            info, _ = completed_metrics(directory)
            if (info.get('iterations'), info.get('resolution_width'), info.get('seed')) != (50000, 1600, 0):
                return False, 'MLP protocol is incomplete: ' + str(directory)
            if directory != baseline and not {'train', 'render', 'metrics'}.issubset(info.get('completed_steps', [])):
                return False, 'MLP postprocessing is incomplete: ' + str(directory)
            if not frame_outputs_complete(directory, 50):
                return False, 'MLP GT/render images are incomplete: ' + str(directory)
        except (OSError, ValueError, KeyError) as error:
            return False, 'MLP result not verified: ' + str(error)
    return True, 'All original MLP training, rendering and metrics are complete'


def initialize():
    REPORTS.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
    items = prepare()
    plan = {'scene': 'rubber_duck', 'model': 'original PIDG baseline, width256/depth0',
            'physics_extensions': 'off', 'mask_type': 'remove whole training frames from the training manifest',
            'drop_percentages': [0, *RATIOS], 'mask_seed': MASK_SEED, 'nested_random_subsets': True,
            'long_protocol': {'iterations': 50000, 'resolution_width': 1600, 'seed': 0},
            'reference': str(BASELINE), 'dependency': str(MLP_RUNS / 'controller.json'),
            'selection': 'original official-test best PSNR; dropped-frame GT never selects snapshots',
            'evaluation': ['unchanged 50 official test views', 'withheld training-frame GT in separate output directories'],
            'jobs': [{k: v for k, v in item.items() if k not in ('train', 'test', 'heldout')} for item in items],
            'limitations': ['one scene, one training seed and one mask seed',
                            'native PIDG CMR queries use fid/train_count; this denominator changes with retained count',
                            'the fixed 50k iteration budget gives more image reuses at high dropout',
                            'baseline best selection uses official test PSNR; this is not leakage-free model selection',
                            'the shared canonical point cloud is retained; the experiment measures missing RGB-frame robustness']}
    save_frozen_json(REPORTS / 'plan.json', plan)
    save_frozen_json(REPORTS / 'source_freeze.json', current_freeze())
    return items


def current_freeze():
    return {'baseline_runtime': runtime_hashes(REPO),
            'original_dataset': fixture_hashes(SOURCE),
            'helpers': {str(path.relative_to(ROOT)): sha256(path) for path in
                        [Path(__file__), HERE / 'prepare.py', BASE_CONFIG,
                         ROOT / 'scripts/run_constitutive_experiment.py',
                         ROOT / 'scripts/pidg_variant_entrypoint.py',
                         ROOT / 'reproduction/scripts/training_checkpoint.py']},
            'reference_results_sha256': sha256(BASELINE / 'results.json')}


def check_freeze():
    if current_freeze() != read_json(REPORTS / 'source_freeze.json'):
        raise RuntimeError('Frozen baseline source, helpers, reference or original dataset changed')


def report():
    controller = read_json(RUNS / 'controller.json') if (RUNS / 'controller.json').exists() else {}
    _, base_metrics = completed_metrics(BASELINE)
    rows = [{'drop_percent': 0, 'retained': 150, 'removed': 0, 'status': 'completed (reused)',
             'iteration': 50000, 'official_metrics': base_metrics, 'masked_holdout_metrics': None,
             'run': str(BASELINE)}]
    for percent in RATIOS:
        directory = RUNS / f'long_drop{percent:02d}_seed{MASK_SEED}' / 'rubber_duck'
        info = read_json(directory / 'reproduction.json') if (directory / 'reproduction.json').exists() else {}
        progress = parse_progress(tail(directory / 'train.log'))
        complete = info.get('status') == 'completed'
        rows.append({'drop_percent': percent, 'retained': 150 - round(150 * percent / 100),
                     'removed': round(150 * percent / 100), 'status': info.get('status', 'queued'),
                     'iteration': 50000 if complete else progress['iteration'],
                     'official_metrics': info.get('metrics', {}).get('ours_best') if complete else None,
                     'masked_holdout_metrics': info.get('masked_holdout_metrics') if complete else None,
                     'run': str(directory)})
    snapshot = {'updated_utc': now(), 'refresh_seconds': 5, 'controller': controller,
                'controller_alive': alive(controller.get('pid')), 'rows': rows}
    write_json(REPORTS / 'results.json', snapshot)
    lines = ['# Baseline PIDG: random missing training frames', '',
             'Queued after the MLP capacity queue AND its rendering/metrics. Whole-frame dropout, not pixel masking.',
             'Rubber Duck; baseline width256/depth0; 50k iterations; width1600; training seed0; mask seed0.', '',
             '| Dropped | Kept / 150 | Withheld GT | Status | Iteration | Official-test PSNR | Delta vs 0% | Withheld-frame PSNR |',
             '|---:|---:|---:|---|---:|---:|---:|---:|']
    fmt = lambda value: '-' if value is None else f'{value:.6f}'
    table_rows = []
    csv_lines = ['drop_percent,retained,removed,status,iteration,official_PSNR,official_SSIM,official_LPIPS,official_MS_SSIM,heldout_PSNR,heldout_SSIM,heldout_LPIPS,heldout_MS_SSIM']
    for row in rows:
        official, heldout = row['official_metrics'] or {}, row['masked_holdout_metrics'] or {}
        psnr, missing_psnr = official.get('PSNR'), heldout.get('PSNR')
        delta = None if psnr is None else psnr - base_metrics['PSNR']
        cells = [f"{row['drop_percent']}%", row['retained'], row['removed'], row['status'], row['iteration'] or '-',
                 fmt(psnr), fmt(delta), fmt(missing_psnr)]
        lines.append('| ' + ' | '.join(map(str, cells)) + ' |')
        table_rows.append('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in cells) + '</tr>')
        csv_lines.append(','.join(map(str, [row['drop_percent'], row['retained'], row['removed'], row['status'], row['iteration'] or '',
                             *[official.get(k, '') for k in ('PSNR', 'SSIM', 'LPIPS', 'MS-SSIM')],
                             *[heldout.get(k, '') for k in ('PSNR', 'SSIM', 'LPIPS', 'MS-SSIM')]])))
    lines += ['', 'The official 50-view test set is identical at every ratio. Dropped training frames are evaluated only after fitting.',
              'Withheld subsets are nested but not identical across ratios; use the fixed official test set for cross-ratio comparisons.',
              'Original timestamps, poses and canonical point cloud remain fixed. Original input files are never deleted or masked in place.',
              'Native PIDG time normalization uses the retained train_count; no physics correction is silently introduced.',
              '0% is reused. Best snapshots use the historical test-selected protocol; single-seed results are exploratory.',
              '', '[Five-second monitor](monitor.html)', '', 'Controller: ' + str(controller.get('status', 'not started')),
              'Dependency: ' + controller.get('dependency_reason', 'not checked'), '', 'Updated UTC: ' + snapshot['updated_utc']]
    write_text_atomic(REPORTS / 'results.md', '\n'.join(lines) + '\n')
    write_text_atomic(REPORTS / 'results.csv', '\n'.join(csv_lines) + '\n')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>PIDG missing-frame queue</title>
<style>body{font:16px system-ui;margin:32px;color:#153149;background:#f3f6f9}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border-bottom:1px solid #d8e2ea;text-align:left}</style>
<h1>Baseline PIDG: random missing training frames</h1><p>0%, 10%, 30%, 50%, 70%, 90% dropout. Whole frames; original baseline. Refresh: 5 seconds.</p>'''
    page += '<p>Status: ' + html.escape(str(controller.get('status', 'not started'))) + '</p><p>' + html.escape(controller.get('dependency_reason', '')) + '</p>'
    page += '<p>Current job: ' + html.escape(controller.get('current_job', '-')) + '</p>'
    page += '<table><tr>' + ''.join('<th>' + x + '</th>' for x in ['Dropped', 'Kept', 'Withheld', 'Status', 'Iteration', 'Official PSNR', 'Delta', 'Withheld PSNR']) + '</tr>' + ''.join(table_rows) + '</table>'
    page += '<p>Official test: fixed 50 views. Withheld GT never enters training or snapshot selection.</p><small>Updated UTC: ' + snapshot['updated_utc'] + '</small></html>'
    write_text_atomic(REPORTS / 'monitor.html', page)
    return snapshot


def make_commands(directory, dataset, heldout_dataset, smoke=False, checkpoint=None):
    prefix = [sys.executable, '-u', str(ROOT / 'scripts/pidg_variant_entrypoint.py'), '--repo', str(REPO), '--seed', '0']
    config = directory / 'baseline_config.py'
    args = ['-s', str(dataset), '-m', str(directory), '--conf', str(config), '--load2gpu_on_the_fly', '--resolution', '400' if smoke else '1600']
    train = [*prefix, 'train_pidg.py', *args, '--iterations', '300' if smoke else '50000', '--checkpoint_interval',
             '300' if smoke else '1000', '--save_image_dir', str(directory / 'diagnostics')]
    if smoke:
        train += ['--warm_up', '0', '--densify_until_iter', '0', '--test_iterations', '300', '--save_iterations', '300']
    if checkpoint is not None:
        train += ['--resume_checkpoint', str(checkpoint)]
    heldout_run = directory / 'masked_holdout'
    heldout_args = ['-s', str(heldout_dataset), '-m', str(heldout_run), '--conf', str(config),
                    '--load2gpu_on_the_fly', '--resolution', '400' if smoke else '1600']
    return [('train', train), ('render', [*prefix, 'render.py', *args, '--iteration', 'best', '--skip_train']),
            ('metrics', [*prefix, 'metrics.py', '-m', str(directory)]),
            ('heldout_render', [*prefix, 'render.py', *heldout_args, '--iteration', 'best', '--skip_train']),
            ('heldout_metrics', [*prefix, 'metrics.py', '-m', str(heldout_run)])]


def run_one(item, smoke=False, resume=False):
    check_freeze()
    name = item['id']; split_root = DATA / name
    dataset, heldout_dataset = split_root / 'dataset', split_root / 'heldout_dataset'
    directory = RUNS / (('smoke_' if smoke else 'long_') + name) / 'rubber_duck'
    expected = {'iterations': 300 if smoke else 50000, 'resolution_width': 400 if smoke else 1600, 'seed': 0,
                'smoke_only': smoke, 'profile': 'random_frame_dropout_smoke' if smoke else 'random_frame_dropout_long',
                'scene': 'rubber_duck', 'config': {'deformation_gradient': False, 'constitutive': False,
                    'structured_force': False, 'learn_material_parameters': False, 'network_args': {'width': 256, 'depth': 0, 'directional': True}},
                'drop_percent': item['drop_percent'], 'retained_train_count': item['retained_count'], 'mask_seed': MASK_SEED,
                'source_hashes': runtime_hashes(REPO), 'dataset_hashes': fixture_hashes(dataset),
                'heldout_dataset_hashes': fixture_hashes(heldout_dataset),
                'split_manifest_sha256': sha256(split_root / 'split_manifest.json'),
                'runner_sha256': sha256(__file__), 'prepare_sha256': sha256(HERE / 'prepare.py'),
                'queue_freeze_sha256': sha256(REPORTS / 'source_freeze.json'),
                'checkpoint_helper_sha256': sha256(ROOT / 'reproduction/scripts/training_checkpoint.py'),
                'entrypoint_sha256': sha256(ROOT / 'scripts/pidg_variant_entrypoint.py')}
    checkpoint = None
    if directory.exists():
        if not resume:
            raise FileExistsError('Existing run preserved; use --resume: ' + str(directory))
        info = read_json(directory / 'reproduction.json')
        if any(info.get(k) != v for k, v in expected.items()):
            raise ValueError('Resume protocol/config/source/input mismatch')
        if info['status'] == 'completed':
            completed_metrics(directory); return
        if 'train' not in info['completed_steps']:
            execution.verified_checkpoint(directory)
            checkpoint = directory / 'checkpoints/resume.pt'
    else:
        directory.mkdir(parents=True)
        shutil.copyfile(BASE_CONFIG, directory / 'baseline_config.py')
        info = {**expected, 'experiment': 'PIDG_BASELINE_FRAME_DROPOUT', 'configuration_id': name,
                'commands': [], 'completed_steps': [], 'started_utc': now(),
                'base_config_sha256': sha256(BASE_CONFIG)}
    env = os.environ.copy(); env.pop('PIDG_CONSTITUTIVE_CONFIG', None)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    env['PIDG_HASH_SOURCE_DIR'] = str(REPO / 'hashencoder'); env['PIDG_HASH_BUILD_DIR'] = str(REPO / 'tmp_build')
    try:
        for step, command in make_commands(directory, dataset, heldout_dataset, smoke, checkpoint):
            if step in info['completed_steps']:
                continue
            if (RUNS / 'stop_queue.request').exists():
                raise SystemExit(75)
            if step == 'heldout_render':
                # Read-only snapshot links; heldout rendering cannot overwrite official test images.
                target = directory / 'masked_holdout'; target.mkdir(exist_ok=True)
                for folder in ('point_cloud', 'deform'):
                    link_directory(target / folder, directory / folder)
                shutil.copyfile(directory / 'cfg_args', target / 'cfg_args')
            info['status'] = {'train': 'training', 'render': 'rendering', 'metrics': 'evaluating',
                              'heldout_render': 'rendering_masked_holdout', 'heldout_metrics': 'evaluating_masked_holdout'}[step]
            write_json(directory / 'reproduction.json', info)
            code = execution.execute(command, directory, step, REPO, env, info)
            if code == 75:
                raise SystemExit(75)
            if code:
                raise RuntimeError(f'{step} failed ({code}); see {directory / step}.log')
            info['completed_steps'].append(step)
            write_json(directory / 'reproduction.json', info)
        assert runtime_hashes(REPO) == expected['source_hashes']
        assert fixture_hashes(dataset) == expected['dataset_hashes']
        assert fixture_hashes(heldout_dataset) == expected['heldout_dataset_hashes']
        assert sha256(directory / 'baseline_config.py') == sha256(BASE_CONFIG) == info['base_config_sha256']
        assert execution.verified_checkpoint(directory)['iteration'] == expected['iterations']
        training_log = (directory / 'train.log').read_text(encoding='utf-8', errors='replace')
        assert 'Training complete.' in training_log
        assert not re.search(r'(?:Loss=|loss = )\s*(?:nan|inf)', training_log, re.I)
        official = read_json(directory / 'results.json')
        masked = read_json(directory / 'masked_holdout/results.json')['ours_best']
        for values in (official['ours_best'], masked):
            assert all(math.isfinite(values[k]) for k in ('PSNR', 'SSIM', 'LPIPS', 'MS-SSIM'))
        assert frame_outputs_complete(directory, 50)
        assert frame_outputs_complete(directory / 'masked_holdout', item['dropped_count'])
        info.update(status='completed', metrics=official, masked_holdout_metrics=masked, finished_utc=now())
        write_json(directory / 'reproduction.json', info); completed_metrics(directory)
    except SystemExit:
        info['status'] = 'saved_stopped'; write_json(directory / 'reproduction.json', info); raise
    except Exception as error:
        info.update(status='failed', error=str(error)); write_json(directory / 'reproduction.json', info); raise


def work(resume=False):
    items = initialize()
    if (RUNS / 'stop_queue.request').exists():
        raise RuntimeError('Saved pause request remains present')
    import run_constitutive_matrix as locks
    locks.RUNS = RUNS
    controller = {'pid': os.getpid(), 'status': 'waiting_for_mlp', 'started_utc': now(),
                  'current_job': 'dependency', 'completed_jobs': [], 'runner_sha256': sha256(__file__)}
    with locks.controller_lock():
        write_json(RUNS / 'controller.json', controller)
        try:
            while True:
                ready, reason = dependency_ready()
                controller.update(dependency_reason=reason, dependency_last_checked_utc=now())
                if ready:
                    controller['dependency_satisfied_utc'] = now(); break
                controller['status'] = 'waiting_for_mlp'
                write_json(RUNS / 'controller.json', controller); report()
                if (RUNS / 'stop_queue.request').exists():
                    raise SystemExit(75)
                time.sleep(5)
            controller['status'] = 'running'
            write_json(RUNS / 'controller.json', controller)
            stop = threading.Event()
            def refresh():
                while not stop.wait(5):
                    try: report()
                    except Exception as error: print('Monitor:', error, flush=True)
            thread = threading.Thread(target=refresh, daemon=True); thread.start()
            try:
                controller['current_job'] = 'smoke_drop90_seed0'
                write_json(RUNS / 'controller.json', controller)
                run_one(items[-1], smoke=True, resume=resume)
                controller['completed_jobs'].append(controller['current_job'])
                for item in items:
                    if shutil.disk_usage(ROOT).free < 15 * 1024 ** 3:
                        raise RuntimeError('Less than 15 GiB free')
                    controller['current_job'] = 'long_' + item['id']
                    write_json(RUNS / 'controller.json', controller)
                    run_one(item, resume=resume)
                    controller['completed_jobs'].append(controller['current_job'])
                controller['status'] = 'completed'
            finally:
                stop.set(); thread.join(timeout=6)
        except SystemExit as error:
            controller['status'] = 'saved_stopped' if error.code == 75 else 'failed'
            if error.code != 75: raise
        except Exception as error:
            controller.update(status='failed', error=str(error)); raise
        finally:
            controller['updated_utc'] = now(); write_json(RUNS / 'controller.json', controller); report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--report', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.prepare or args.report:
        initialize(); report()
    else:
        work(args.resume)
