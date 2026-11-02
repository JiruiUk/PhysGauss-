"""Two matched C1 3k->50k arms; the only treatment flag is observed_flow_enabled."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.color_repair import run as parent
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.c1_flow_ablation.prepare import prepare

HERE = Path(__file__).parent
RUNS = ROOT/'runs/pidg_c1_flow_ablation'
REPORTS = ROOT/'experiments/c1_flow_ablation'
OUT = ROOT/'output/c1_flow_ablation'
NAMES = ['O0_C1_flow_off', 'O1_C1_flow_on']


def configuration(name, smoke=False, off=False):
    config, text = parent.configuration('C1_repaired', smoke, off)
    config.update(observed_flow_enabled=name == NAMES[1] and not off,
        observed_flow_weight=.01, flow_start=20 if smoke else 3000,
        flow_ramp=10 if smoke else 2000, flow_interval=10 if smoke else 100,
        record_sampling=True, audit_flow_gradients=smoke, velocity_source='independent')
    return config, text


def wire(repo):
    base.RUNS, base.REPORTS, base.assert_frozen = RUNS, REPORTS, assert_frozen
    engine.RUNS, engine.REPORTS, engine.repo_global, engine.assert_frozen = RUNS, REPORTS, repo, assert_frozen


def source_prefix(smoke=False):
    return RUNS/'_smoke/shared_prefix/rubber_duck' if smoke else parent.RUNS/'shared_prefix/rubber_duck'


def freeze(repo):
    from variants.unified_flow.prepare_data import CACHE
    paths = [*HERE.glob('*.py'), ROOT/'tests/test_c1_flow_ablation.py',
        ROOT/'scripts/pidg_variant_entrypoint.py', ROOT/'reproduction/scripts/training_checkpoint.py']
    for directory in ['metric_tuning', 'kinematic_tuning', 'kinematic_resplit', 'v3_iso_scratch',
            'coverage_tuning', 'color_repair', 'centroid_tuning']:
        paths += list((ROOT/'variants'/directory).glob('*.py'))
    paths += [ROOT/p for p in ['variants/flow_comparison/temporal.py', 'variants/flow_comparison/geometry.py',
        'variants/flow_comparison/flow.py', 'variants/unified_flow/flow.py',
        'variants/unified_flow/camera_contract.py', 'variants/unified_flow/prepare_data.py']]
    prefix = source_prefix()
    paths += [prefix/p for p in ['model_config.py', 'metric_config.json', 'checkpoints/resume.json',
        'checkpoints/resume.metrics.json']]
    # Actual immutable image and flow data bytes, not only their file lists.
    for mode in ('training', 'final_test'):
        paths += [p for p in (base.DATA/mode).rglob('*') if p.is_file()]
    paths += [p for p in CACHE.iterdir() if p.is_file()]
    val_cache = ROOT/'data/derived/flow_comparison_validation_eval_v2'
    paths += [p for p in val_cache.iterdir() if p.is_file()]
    return {'code': engine.runtime_hashes(repo), 'C1_parent_code': engine.runtime_hashes(parent.HERE/'code'),
        'files': {str(p.relative_to(ROOT)): engine.sha256(p) for p in sorted(set(paths))},
        'full_prefix_sha256': engine.sha256(source_prefix()/'checkpoints/resume.pt')}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json') != freeze(repo):
        raise ValueError('Frozen C1 ablation source, checkpoints or data changed')


def initialize(name, smoke=False, off=False):
    directory = RUNS/('_smoke' if smoke else '')/name/'rubber_duck'
    config, text = configuration(name, smoke, off)
    if (directory/'reproduction.json').exists():
        assert engine.read_json(directory/'metric_config.json') == config
        assert (directory/'model_config.py').read_text() == text
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    engine.write_json(directory/'metric_config.json', config)
    engine.write_text_atomic(directory/'model_config.py', text)
    engine.write_json(directory/'reproduction.json', {'id': name, 'commands': [], 'status': 'initialized',
        'target_iterations': 20 if off else 100 if smoke else 50000, 'resolution_width': 400 if smoke else 1600,
        'source_freeze_sha256': engine.sha256(REPORTS/'source_freeze.json'), 'warm_start_final_model': False})
    return directory


def branch(destination, smoke=False):
    source = source_prefix(smoke); boundary = 20 if smoke else 3000
    if (destination/'branch.json').exists():
        receipt = engine.read_json(destination/'branch.json')
        assert receipt['source'] == str(source) and receipt['iteration'] == boundary
        return
    if base.progress(destination) > 0:
        raise ValueError('Unreceipted branch exists')
    assert parent.step(source) == boundary
    assert (source/'model_config.py').read_text() == (destination/'model_config.py').read_text()
    digest = engine.sha256(source/'checkpoints/resume.pt')
    assert digest == engine.read_json(source/'checkpoints/resume.json')['sha256']
    original = engine.read_json(source/'metric_config.json')
    current = engine.read_json(destination/'metric_config.json')
    changed = {k for k in set(original)|set(current) if original.get(k) != current.get(k)}
    assert changed <= {'coverage_density', 'observed_flow_enabled', 'observed_flow_weight', 'flow_start',
        'flow_ramp', 'flow_interval', 'record_sampling', 'audit_flow_gradients', 'velocity_source'}
    for folder in ('point_cloud', 'deform', 'checkpoints'):
        shutil.copytree(source/folder, destination/folder, dirs_exist_ok=True)
    for filename in ('metric_best.json', 'metric_history.jsonl'):
        shutil.copyfile(source/filename, destination/filename)
    sidecar = destination/'checkpoints/resume.metrics.json'; saved = engine.read_json(sidecar)
    assert saved['config'] == original
    saved['config'] = current; engine.write_json(sidecar, saved)
    assert engine.sha256(destination/'checkpoints/resume.pt') == digest
    engine.write_json(destination/'branch.json', {'source': str(source), 'destination': str(destination),
        'iteration': boundary, 'identical_full_checkpoint_sha256': digest, 'changed_config_keys': sorted(changed),
        'tensor_optimizer_RNG_bytes_unchanged': True, 'created_utc': engine.now()})


def preflight(repo, deadline):
    proof = engine.read_json(REPORTS/'preflight.json') if (REPORTS/'preflight.json').exists() else {}
    if proof.get('source_freeze_sha256') == engine.sha256(REPORTS/'source_freeze.json'):
        return
    env = engine.env_for(RUNS); env['PIDG_COLOR_REPAIR_GPU_TEST'] = '1'; env['PIDG_UNIFIED_GPU_TEST'] = '1'
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_c1_flow_ablation.py',
        'tests/test_unified_flow_geometry.py', 'tests/test_color_repair_gpu.py'], cwd=ROOT, env=env, capture_output=True, text=True)
    engine.write_text_atomic(REPORTS/'tests.log', result.stdout+result.stderr)
    if result.returncode:
        raise ValueError('C1 flow tests failed')
    off = initialize('all_flags_off', True, True)
    if parent.step(off) < 20:
        assert base.train(off, repo, deadline, True, off=True) == 0
    common = initialize('shared_prefix', True)
    if parent.step(common) < 20:
        assert base.train(common, repo, deadline, True, limit=20) == 75
    assert parent.step(common) == 20
    for name in NAMES:
        d = initialize(name, True); branch(d, True)
        if name == NAMES[1] and parent.step(d) < 70:
            assert base.train(d, repo, deadline, True, limit=70) == 75
        if parent.step(d) < 100:
            assert base.train(d, repo, deadline, True) == 0
    on = RUNS/'_smoke'/NAMES[1]/'rubber_duck'
    assert 'FULL_CHECKPOINT_RESTORED completed=70 next=71' in (on/'train.log').read_text()
    assert len((on/'flow_gradients.jsonl').read_text().splitlines()) >= 3
    off_arm = RUNS/'_smoke'/NAMES[0]/'rubber_duck'
    assert not (off_arm/'observed_flow.jsonl').exists()
    # Isolate the new loss from image/momentum and prove an actual optimizer update.
    result = subprocess.run([sys.executable, '-u', '-m', 'variants.c1_flow_ablation.verify_update'],
        cwd=ROOT, env=env, capture_output=True, text=True)
    engine.write_text_atomic(REPORTS/'optimizer_probe.log', result.stdout+result.stderr)
    if result.returncode:
        raise ValueError('C1 flow optimizer probe failed')
    engine.write_json(REPORTS/'preflight.json', {'status': 'passed', 'tests': result.returncode == 0,
        'smoke_steps_each': 100, 'flow_on_resume': [70, 100], 'all_flags_off_steps': 20,
        'only_treatment_flag': 'observed_flow_enabled', 'actual_flow_optimizer_update': True,
        'source_freeze_sha256': engine.sha256(REPORTS/'source_freeze.json'), 'completed_utc': engine.now()})


def report():
    rows = []
    for name in NAMES:
        d = RUNS/name/'rubber_duck'; h = base.history(d)
        rows.append({'model': name, 'iteration_saved': parent.step(d),
            'validation': h[-1] if h else None,
            'test': engine.read_json(d/'results.json') if (d/'results.json').exists() else None})
    controller = engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    engine.write_json(REPORTS/'progress.json', {'updated_utc': engine.now(), 'controller': controller, 'models': rows})


def plan():
    return {'baseline': 'C1_repaired', 'selection_reason': 'best historical C-series validation PSNR39.6648946; test40.3721275 descriptive only',
        'arms': {n: configuration(n)[0] for n in NAMES}, 'only_treatment_flag': 'observed_flow_enabled',
        'same_model_config': True, 'same_full_state_prefix': str(source_prefix()), 'prefix_steps': 3000,
        'total_steps_each': 50000, 'additional_steps_each': 47000, 'total_new_full_training_steps': 94000,
        'seed': 0, 'scene': 'rubber_duck', 'split': [120, 30, 50], 'RGB_width': 1600, 'flow_resolution': [400, 225],
        'loss': '0.01*ramp*0.5*(raw geometric optical-flow Huber + independent-velocity-advected optical-flow Huber), every100 after3k, ramp2k',
        'unchanged': ['C1 architecture and independent velocity/stress', 'original C1 Cauchy residual and sample schedule',
            'C1 frame-index shared time', 'RGB/SSIM/LPIPS, mask and foreground losses', 'densification and optimizers'],
        'no_new_parameters_or_initialization': True, 'new_guard_or_coherent_momentum': False,
        'selection': 'both arms finish; lock validation-PSNR best plus final50k before test and validation-motion diagnostics',
        'primary_contrast': 'new O1 minus new O0; historical C1 only descriptive reference',
        'prospective_benefit': {'validation_PSNR_min_delta_dB': -.2, 'validation_LPIPS_max_relative_increase': .05,
            'validation_velocity_EPE_min_relative_reduction': .10, 'validation_geometry_EPE_max_relative_increase': .05},
        'report_all_axes': True, 'historical_test_exposure': True, 'single_scene_single_seed': True,
        'GPU_nondeterminism': True, 'flow_is_estimated_2D_not_physical_GT': True,
        'no_automatic_additional_candidates': True}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--hours', type=float, default=16.)
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    repo = prepare(); wire(repo)
    from variants.unified_flow.prepare_data import prepare as verify_raw_flow
    verify_raw_flow()
    if (REPORTS/'source_freeze.json').exists(): assert_frozen(repo)
    else: engine.write_json(REPORTS/'source_freeze.json', freeze(repo))
    protocol = plan()
    if (REPORTS/'plan.json').exists(): assert engine.read_json(REPORTS/'plan.json') == protocol
    else: engine.write_json(REPORTS/'plan.json', protocol)
    deadline = datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state = {'pid': os.getpid(), 'status': 'preflight', 'started_utc': engine.now(), 'deadline_utc': deadline.isoformat()}
    with engine.lock():
        if (RUNS/'stop_queue.request').exists(): raise ValueError('Explicit stop request remains')
        engine.write_json(RUNS/'controller.json', state)
        os.environ['PIDG_EXPORT_ALPHA'] = '1'
        try:
            preflight(repo, deadline)
            if args.preflight_only:
                state['status'] = 'preflight_passed'; return
            state['status'] = 'training'; engine.write_json(RUNS/'controller.json', state)
            for name in NAMES:
                d = initialize(name); branch(d)
            if all(parent.step(RUNS/n/'rubber_duck') == 3000 for n in NAMES):
                assert engine.sha256(RUNS/NAMES[0]/'rubber_duck/checkpoints/resume.pt') == engine.sha256(RUNS/NAMES[1]/'rubber_duck/checkpoints/resume.pt')
            for name in NAMES:
                d = RUNS/name/'rubber_duck'; report()
                if parent.step(d) < 50000 and base.train(d, repo, deadline) == 75:
                    raise SystemExit(75)
            lock = RUNS/'selection_lock.json'
            selections = {n: engine.read_json(RUNS/n/'rubber_duck/metric_best.json')['PSNR'] for n in NAMES}
            content = {'selection_split': 'validation', 'models': selections}
            if lock.exists():
                saved = engine.read_json(lock)
                assert {k:saved[k] for k in content} == content
            else: engine.write_json(lock, {**content, 'locked_utc': engine.now()})
            state['status'] = 'final_test'; engine.write_json(RUNS/'controller.json', state)
            for name in NAMES:
                d = RUNS/name/'rubber_duck'
                base.render(d, repo, deadline, tags=['best', '50000'])
            state['status'] = 'postprocess'; engine.write_json(RUNS/'controller.json', state)
            from variants.c1_flow_ablation.results import finalize
            finalize()
            assert_frozen(repo)
            state['status'] = 'completed'
        except SystemExit as error:
            state['status'] = 'saved_stopped' if error.code == 75 else 'failed'
            if error.code != 75: raise
        except Exception as error:
            state.update(status='failed', error=str(error)); raise
        finally:
            state['updated_utc'] = engine.now(); engine.write_json(RUNS/'controller.json', state); report()


if __name__ == '__main__': main()
