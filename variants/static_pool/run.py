"""Matched full-state R0 arms, guarded preflight, immutable inputs and source."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import json
import os
import shutil
import subprocess
import sys
import torch
from variants.metric_tuning import run as engine
from .prepare import prepare

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'experiments/static_pool'
RUNS = ROOT/'runs/pidg_static_pool'
DATA = ROOT/'data/derived/kinematic_resplit/training'
NAMES = ['H0_dynamic_control', 'H1_static_pool']


def plan(): return engine.read_json(REPORT/'plan.json')
def source(): return ROOT/plan()['source_directory']
def step(d):
    p = d/'checkpoints/resume.json'
    return engine.read_json(p)['iteration'] if p.exists() else -1


def wire(repo):
    engine.RUNS, engine.REPORTS, engine.repo_global = RUNS, REPORT, repo


def configuration(on):
    b = engine.read_json(REPORT/'binding.json')
    c = engine.read_json(source()/'metric_config.json')
    assert not c['coverage_density'] and c['affine_weight'] == 0
    c.update(hard_static_pool=on, static_pool_binding=b['path'], static_pool_binding_sha256=b['sha256'],
             static_pool_source_sha256=plan()['source_checkpoint_sha256'], record_static_pool_sampling=True)
    return c


def branch(name, on, smoke=False):
    d = RUNS/('_smoke' if smoke else 'pilot')/name/'rubber_duck'
    c = configuration(on)
    if (d/'branch.json').exists():
        assert engine.read_json(d/'metric_config.json') == c
        return d
    assert not (d/'checkpoints/resume.pt').exists(), 'Unreceipted branch state'
    d.mkdir(parents=True, exist_ok=True)
    engine.write_json(d/'metric_config.json', c)
    shutil.copyfile(source()/'model_config.py', d/'model_config.py')
    (d/'checkpoints').mkdir(exist_ok=True)
    for p in (source()/'checkpoints').glob('resume.*'):
        shutil.copyfile(p, d/'checkpoints'/p.name)
    digest = engine.sha256(d/'checkpoints/resume.pt')
    assert digest == plan()['source_checkpoint_sha256']
    side = d/'checkpoints/resume.metrics.json'
    s = engine.read_json(side); assert s['config'] == engine.read_json(source()/'metric_config.json')
    s.update(config=c, static_pool_binding_sha256=c['static_pool_binding_sha256'])
    engine.write_json(side, s)
    original_source = Path(engine.read_json(source()/'checkpoints/resume.json')['path']).parents[1]
    engine.write_json(d/'branch.json', {'source': str(original_source), 'destination': str(d),
        'immutable_initialization_source': str(source()),
        'iteration': plan()['source_iteration'], 'identical_full_checkpoint_sha256': digest,
        'full_optimizer_RNG_sampler_preserved': True})
    engine.write_json(d/'reproduction.json', {'id': name, 'commands': [], 'status': 'initialized',
        'seed': plan()['seed'], 'target_iteration': plan()['final_iteration']})
    return d


def train(d, repo, stop, deadline):
    prefix = engine.command_prefix(repo); prefix[-1] = str(plan()['seed'])
    command = [*prefix, 'train_pidg.py', '-s', str(DATA), '-m', str(d), '--conf', str(d/'model_config.py'),
        '--load2gpu_on_the_fly', '--resolution', '1600', '--iterations', '50001', '--checkpoint_interval', '1000',
        '--save_image_dir', str(d/'diagnostics'), '--save_iterations', '15000', '50000',
        '--test_iterations', *map(str, range(5000, 50001, 5000)), '--resume_checkpoint', str(d/'checkpoints/resume.pt')]
    assert engine.sha256(d/'checkpoints/resume.pt') == engine.read_json(d/'checkpoints/resume.json')['sha256']
    env = engine.env_for(d); env['PIDG_TUNING_STOP_AFTER'] = str(stop)
    return engine.execute(command, d, 'long', env, deadline)


def freeze(repo):
    core = ['__init__.py','model.py','render.py','runtime.py','prepare.py','bind.py','probe.py','native_check.py','run.py']
    own = [*(Path(__file__).parent/p for p in core), ROOT/'tests/test_static_pool.py', REPORT/'plan.json', REPORT/'binding.json']
    inherited = ['variants/color_repair/runtime.py', 'variants/color_repair/color.py',
        'variants/kinematic_resplit/checkpoint.py', 'reproduction/scripts/training_checkpoint.py',
        'scripts/pidg_variant_entrypoint.py', 'variants/metric_tuning/run.py']
    inputs = [source()/'checkpoints/resume.pt', DATA/'points3d.ply', DATA/'transforms_train.json',
              DATA/'transforms_test.json', ROOT/'experiments/scratch_integrity/data_binding.json',
              ROOT/'experiments/transfer_integrity/static_image_audit.json']
    b = engine.read_json(ROOT/plan()['binding_metadata']); inputs.append(Path(b['path']))
    value = {'own': {p.relative_to(ROOT).as_posix(): engine.sha256(p) for p in own},
             'inherited': {p: engine.sha256(ROOT/p) for p in inherited},
             'inputs': {p.relative_to(ROOT).as_posix(): engine.sha256(p) for p in inputs},
             'fork': engine.runtime_hashes(repo),
             'C1_parent': engine.runtime_hashes(ROOT/'variants/color_repair/code')}
    path = REPORT/'training_freeze.json'
    if path.exists(): assert engine.read_json(path) == value, 'R0 frozen source/input changed'
    else: engine.write_json(path, value)


def child(module, name):
    result = subprocess.run([sys.executable, '-u', '-m', module], cwd=ROOT,
                            env=engine.env_for(RUNS), capture_output=True, text=True)
    (REPORT/name).write_text(result.stdout+result.stderr, encoding='utf-8')
    assert result.returncode == 0, f'{module} failed: {name}'


def preflight(repo, deadline):
    path = REPORT/'preflight.json'
    if path.exists():
        assert engine.read_json(path)['binding_sha256'] == engine.read_json(REPORT/'binding.json')['sha256']
        return
    child('variants.static_pool.probe', 'optimizer_probe.log')
    baseline = ROOT/'variants/color_repair/code'
    dirs = []
    for name, code in [('flags_off', repo), ('original', baseline), ('original_repeat', baseline)]:
        d = branch(name, False, True); dirs.append(d)
        if step(d) < 15120: assert train(d, code, 15120, deadline) == 75
        assert step(d) == 15120
    states = [torch.load(d/'checkpoints/resume.pt', map_location='cpu', weights_only=False) for d in dirs]
    differences, repeats = {}, {}
    from variants.association_training.run import tensor_equal
    for key in ['gaussian_parameters', 'gaussian_buffers', 'velocity_net', 'grid4d', 'deform']:
        def stats(s):
            ds = []
            for k, v in s[key].items():
                if key == 'gaussian_buffers' and k == 'max_radii2D': continue
                if not v.is_floating_point():
                    assert torch.equal(v, states[1][key][k]); continue
                if v.numel(): ds.append((v.double()-states[1][key][k].double()).flatten())
            delta = torch.cat(ds)
            return {'RMS': float(delta.square().mean().sqrt()), 'max_abs': float(delta.abs().max())}
        differences[key], repeats[key] = stats(states[0]), stats(states[2])
        assert differences[key]['RMS'] <= 3*repeats[key]['RMS']+1e-7, (key, differences[key], repeats[key])
    for k in ['rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda', 'viewpoint_stack']:
        assert tensor_equal(states[0][k], states[1][k]) and tensor_equal(states[2][k], states[1][k]), k
    del states
    enabled = branch('enabled_resume', True, True)
    if step(enabled) < 15060: assert train(enabled, repo, 15060, deadline) == 75
    if step(enabled) < 15120: assert train(enabled, repo, 15120, deadline) == 75
    assert step(enabled) == 15120
    assert 'FULL_CHECKPOINT_RESTORED completed=15060 next=15061' in (enabled/'train.log').read_text()
    child('variants.static_pool.native_check', 'native_check.log')
    native = engine.read_json(REPORT/'native_check.json')
    assert native['passed']
    engine.write_json(path, {'status': 'passed', 'binding_sha256': engine.read_json(REPORT/'binding.json')['sha256'],
        'disabled_updates': 120, 'enabled_updates': 120, 'resume_boundary': [15060,15120],
        'CMR_event_included': True, 'disabled_parameter_RMS': differences, 'original_repeat_RMS': repeats,
        'RNG_sampler_exact_off': True, 'native_render_probe': native,
        'inert_max_radii2D_excluded_from_parameter_gate': 'Densification ended at15k; never used by later optimizer or renderer'})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--hours', type=float, default=12); args = parser.parse_args()
    torch.set_num_threads(4); RUNS.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    repo = prepare(); wire(repo)
    assert engine.sha256(source()/'checkpoints/resume.pt') == plan()['source_checkpoint_sha256']
    deadline = datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state = {'status': 'binding', 'pid': os.getpid(), 'started_utc': engine.now()}
    try:
        engine.write_json(RUNS/'controller.json', state)
        child('variants.static_pool.bind', 'binding.log')
        state['status'] = 'preflight'; engine.write_json(RUNS/'controller.json', state)
        preflight(repo, deadline)
        freeze(repo)
        if args.preflight_only:
            state['status'] = 'preflight_passed'; return
        arms = [branch(n, on) for n, on in zip(NAMES, [False, True])]
        cs = [engine.read_json(d/'metric_config.json') for d in arms]
        assert {k for k in cs[0] if cs[0][k] != cs[1][k]} == {'hard_static_pool'}
        engine.write_json(REPORT/'branch_lock.json', {'source_sha256': plan()['source_checkpoint_sha256'],
            'binding': engine.read_json(REPORT/'binding.json'), 'only_main_flag_differs': True,
            'models': {n: str(d) for n,d in zip(NAMES,arms)}})
        state['status'] = 'training'; engine.write_json(RUNS/'controller.json', state)
        for d in arms:
            if step(d) < 50000: assert train(d, repo, 50000, deadline) == 75
            assert step(d) == 50000; freeze(repo)
        samples = [[json.loads(x) for x in (d/'camera_sampling.jsonl').read_text().splitlines()] for d in arms]
        assert samples[0] == samples[1] and [z['iteration'] for z in samples[0]] == list(range(15001,50001))
        state['status'] = 'trained_pending_evaluation'
    except SystemExit as error:
        state.update(status='saved_stopped', exit_code=error.code); raise
    except Exception as error:
        state.update(status='failed', error=str(error)); raise
    finally:
        state['updated_utc'] = engine.now(); engine.write_json(RUNS/'controller.json', state)


if __name__ == '__main__': main()
