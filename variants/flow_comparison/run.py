"""Sequential matched-budget optical-flow / structural-velocity experiment."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.monotonic_training import run as previous
from .prepare import prepare

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
REPORTS = ROOT/'experiments/flow_comparison'
RUNS = ROOT/'runs/pidg_flow_comparison'
PREFIX = previous.PREFIX
NAMES = ['K1_flow_independent', 'K2_flow_trajectory']


def config(name, smoke=False, off=False):
    c = engine.read_json(previous.REFERENCE/'metric_config.json')
    c.update(observed_flow=True, material_momentum=True, velocity_source='trajectory' if name == NAMES[1] else 'independent',
        velocity_clock_mode='fixed_monotonic', kinematic_weight=0., velocity_dt=.001, acceleration_dt=1/199,
        flow_start=3000, flow_ramp=2000, momentum_start=8000, momentum_ramp=4000, momentum_samples=64,
        momentum_neighbors=12, audit_parameters=False)
    text = (PREFIX/'model_config.py').read_text(encoding='utf8')+'\nuse_flow = True\n'
    if smoke:
        c.update(engineering_smoke=True, audit_parameters=True, flow_start=20, flow_ramp=10,
            momentum_start=30, momentum_ramp=10, momentum_samples=12, momentum_neighbors=12,
            integrity_coverage_start=20, integrity_strain_start=30, integrity_ramp=10,
            integrity_strain_interval=5, integrity_screen_iteration=100, mask_start=20, mask_ramp=10)
        text += '\nwarm_up = 0\ndensify_until_iter = 0\n'
    if off:
        c.update(enabled=False, observed_flow=False, material_momentum=False, velocity_source='independent',
            velocity_clock_mode='legacy', integrity_enabled=False)
        text += '\nuse_flow = False\n'
    return c, text


def initialize(name, smoke=False, off=False):
    d = RUNS/('_smoke' if smoke else '')/('flags_off' if off else name)/'rubber_duck'
    c, text = config(name, smoke, off)
    d.mkdir(parents=True, exist_ok=True)
    if (d/'metric_config.json').exists():
        if engine.read_json(d/'metric_config.json') != c or (d/'model_config.py').read_text(encoding='utf8') != text:
            raise ValueError('Existing experiment configuration changed')
        return d
    engine.write_json(d/'metric_config.json', c)
    engine.write_text_atomic(d/'model_config.py', text)
    engine.write_json(d/'reproduction.json', {'id': name, 'status': 'initialized', 'commands': [],
        'train': 120, 'validation': 30, 'test': 50, 'selection': 'validation only',
        'seed': 0, 'target_iterations': 20 if off else 100 if smoke else 50000})
    return d


def freeze(repo):
    p = previous.freeze(ROOT/'variants/monotonic_training/code')
    files = [*HERE.glob('*.py'), *ROOT.glob('tests/test_flow_comparison*.py'), REPORTS/'plan.json',
        REPORTS/'flow_model_lock.json', REPORTS/'flow_data_manifest.json', HERE/'source_manifest.json']
    return {'parent': p, 'code': engine.runtime_hashes(repo),
        'helpers': {str(f.relative_to(ROOT)): engine.sha256(f) for f in files}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json') != freeze(repo):
        raise ValueError('Frozen code/config/input manifest changed')


def wire(repo):
    engine.RUNS, engine.REPORTS, engine.repo_global, engine.assert_frozen = RUNS, REPORTS, repo, assert_frozen
    base.RUNS, base.REPORTS, base.assert_frozen = RUNS, REPORTS, assert_frozen


def branch(d):
    if (d/'branch.json').exists():
        return
    if (d/'checkpoints/resume.pt').exists():
        raise ValueError('Unreceipted branch')
    metadata = engine.read_json(PREFIX/'checkpoints/resume.json')
    if metadata['iteration'] != 3000 or engine.sha256(PREFIX/'checkpoints/resume.pt') != metadata['sha256']:
        raise ValueError('Wrong shared prefix checkpoint')
    allowed = (PREFIX/'model_config.py').read_text(encoding='utf8')+'\nuse_flow = True\n'
    if (d/'model_config.py').read_text(encoding='utf8') != allowed:
        raise ValueError('Only use_flow may change in the original optimization recipe')
    for folder in ('point_cloud', 'deform', 'checkpoints'):
        shutil.copytree(PREFIX/folder, d/folder, dirs_exist_ok=True)
    for name in ('metric_best.json', 'metric_history.jsonl'):
        shutil.copyfile(PREFIX/name, d/name)
    old = engine.read_json(PREFIX/'metric_config.json'); c = engine.read_json(d/'metric_config.json')
    sidecar = d/'checkpoints/resume.metrics.json'; state = engine.read_json(sidecar)
    if state['config'] != old:
        raise ValueError('Unexpected prefix selection sidecar')
    state['config'] = c; engine.write_json(sidecar, state)
    engine.write_json(d/'branch.json', {'source': str(PREFIX), 'destination': str(d), 'boundary': 3000,
        'identical_full_checkpoint_sha256': metadata['sha256'], 'source_config': old, 'branch_config': c,
        'tensor_optimizer_rng_unchanged_at_branch': True, 'option_change': 'use_flow False -> True; diagnostics relocation',
        'changed_flags': {k: [old.get(k), c.get(k)] for k in set(old)|set(c) if old.get(k) != c.get(k)}})


def preflight(repo, deadline):
    receipt = REPORTS/'preflight.json'
    if receipt.exists() and engine.read_json(receipt).get('status') == 'passed':
        return
    env = engine.env_for(RUNS); env['PIDG_FLOW_GPU_TEST'] = '1'; env['PIDG_MONOTONIC_GPU_TEST'] = '1'
    result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_flow_comparison_data.py',
        'tests/test_flow_comparison_model.py', 'tests/test_monotonic_training.py', '-q',
        '--basetemp', 'tmp/flow_gpu_pytest'], cwd=ROOT, env=env, capture_output=True)
    (REPORTS/'tests.log').write_bytes(result.stdout+result.stderr)
    if result.returncode:
        raise ValueError('Unit / CUDA checks failed; see tests.log')
    audits = {}
    for name in NAMES:
        d = initialize(name, True)
        if previous.step(d) < 70 and base.train(d, repo, deadline, True, limit=70) != 75:
            raise ValueError('Smoke save boundary failed')
        if previous.step(d) < 100 and base.train(d, repo, deadline, True) != 0:
            raise ValueError('Smoke continuation failed')
        if b'FULL_CHECKPOINT_RESTORED completed=70 next=71' not in (d/'train.log').read_bytes():
            raise ValueError('Resume not exercised')
        rows = [json.loads(s) for s in (d/'parameter_gradients.jsonl').read_text().splitlines()]
        if {r['loss'] for r in rows} != {'flow', 'momentum'}:
            raise ValueError('Both observed-flow and physics gradients must be audited')
        if any(not r['finite'] or (r['required'] and r['gradient_l1'] <= 0) for r in rows):
            raise ValueError('Invalid smoke gradients')
        audits[name] = rows
    off = initialize(NAMES[0], True, True)
    if previous.step(off) < 20 and base.train(off, repo, deadline, True, off=True) != 0:
        raise ValueError('Flags-off smoke failed')
    engine.write_json(receipt, {'status': 'passed', 'smoke_steps_per_arm': 100, 'resumed_from': 70,
        'flags_off_steps': 20, 'audits': audits, 'completed_utc': engine.now()})


def monitor(status, active=None, error=None):
    rows = []
    for name in NAMES:
        d = RUNS/name/'rubber_duck'
        history = base.history(d) if d.exists() else []
        rows.append({'id': name, 'iteration': base.progress(d) if d.exists() else 0,
            'latest_validation': history[-1] if history else None})
    state = {'status': status, 'active': active, 'models': rows, 'error': error, 'updated_utc': engine.now()}
    engine.write_json(REPORTS/'progress.json', state)
    import html
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="10"><title>Optical flow comparison</title><style>body{font:18px system-ui;margin:30px;background:#112035;color:#eef4ff}pre{white-space:pre-wrap}</style><h1>光流监督 / 轨迹导出速度</h1><pre>'+html.escape(json.dumps(state, ensure_ascii=False, indent=2))+'</pre>'
    engine.write_text_atomic(REPORTS/'monitor.html', page)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--hours', type=float, default=12); args = parser.parse_args()
    RUNS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    from .prepare_data import prepare as data_prepare
    data_prepare()
    repo = prepare(); wire(repo)
    if (REPORTS/'source_freeze.json').exists():
        assert_frozen(repo)
    else:
        engine.write_json(REPORTS/'source_freeze.json', freeze(repo))
    deadline = datetime.now(timezone.utc)+timedelta(hours=args.hours)
    try:
        monitor('preflight'); preflight(repo, deadline)
        if args.preflight_only:
            monitor('preflight_passed'); return
        stop = threading.Event(); current = [None]
        def refresh():
            while not stop.wait(10):
                monitor('training', current[0])
        thread = threading.Thread(target=refresh, daemon=True); thread.start()
        try:
            for name in NAMES:
                current[0] = name; d = initialize(name); branch(d); monitor('training', name)
                if previous.step(d) < 50000 and base.train(d, repo, deadline) != 0:
                    raise RuntimeError('Training saved before completion')
                if previous.step(d) != 50000:
                    raise ValueError('Incomplete full run')
        finally:
            stop.set(); thread.join()
        # Both training arms finish before either final-test rendering is opened.
        engine.write_json(RUNS/'selection_lock.json', {'locked_utc': engine.now(), 'selection_split': 'validation',
            'models': {name: engine.read_json(RUNS/name/'rubber_duck/metric_best.json')['PSNR'] for name in NAMES},
            'test_opened_before_lock': False, 'historical_test_exposure': True})
        from .export import export, report
        for name in NAMES:
            d = RUNS/name/'rubber_duck'; monitor('rendering', name)
            base.render(d, repo, deadline, split='test', tags=['best', '50000']); export(d)
        report(); monitor('completed')
    except Exception as error:
        monitor('failed', error=repr(error)); raise


if __name__ == '__main__':
    main()
