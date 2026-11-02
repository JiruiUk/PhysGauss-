"""One bounded matched continuation, with original full-state save/resume."""
import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.association_training.prepare import prepare
from variants.association_training.prepare_data import prepare as prepare_data
from variants.association_training.freeze import freeze, core_hashes
from variants.metric_tuning import run as engine

HERE = Path(__file__).parent
RUNS = ROOT / 'runs/pidg_association_training'
REPORT = ROOT / 'experiments/association_training'
SOURCE = ROOT / 'runs/pidg_color_repair/C1_repaired/rubber_duck'
DATA = ROOT / 'data/derived/kinematic_resplit/training'
NAMES = ['T0_C1_continue', 'T1_association']


def configuration(on, binding):
    c = engine.read_json(SOURCE / 'metric_config.json')
    c.update(association_training_enabled=on, association_data=binding['path'],
             association_data_sha256=binding['sha256'], association_source_checkpoint_sha256=binding['source_checkpoint_sha256'],
             association_start=50000, association_interval=20, association_ramp=200,
             association_anchor_weight=.01, association_temporal_weight=.002, record_association_sampling=True)
    return c


def tensor_equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tensor_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(tensor_equal(x, y) for x, y in zip(a, b))
    if hasattr(a, 'shape'):
        import numpy as np
        return bool(np.array_equal(a, b))
    return a == b


def branch(name, target, binding, smoke=False):
    directory = RUNS / ('_smoke' if smoke else 'pilot') / name / 'rubber_duck'
    config = configuration(name == NAMES[1], binding)
    if (directory / 'branch.json').exists():
        assert engine.read_json(directory / 'metric_config.json') == config
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    engine.write_json(directory / 'metric_config.json', config)
    shutil.copyfile(SOURCE / 'model_config.py', directory / 'model_config.py')
    common = RUNS / 'branch_states' / str(target) / 'resume.pt'
    if not common.exists():
        original = torch.load(SOURCE / 'checkpoints/resume.pt', map_location='cpu', weights_only=False)
        assert engine.sha256(SOURCE / 'checkpoints/resume.pt') == binding['source_checkpoint_sha256']
        changed = dict(original)
        changed['opt'] = dict(original['opt'], iterations=target + 1)
        changed['testing_iterations'] = [target] if smoke else [50500, target]
        changed['saving_iterations'] = [target]
        common.parent.mkdir(parents=True, exist_ok=True)
        torch.save(changed, common)
        restored = torch.load(common, map_location='cpu', weights_only=False)
        fields = [k for k in original if k not in ['opt', 'testing_iterations', 'saving_iterations']]
        assert all(tensor_equal(original[k], restored[k]) for k in fields)
        assert {k for k in original['opt'] if original['opt'][k] != restored['opt'][k]} == {'iterations'}
        engine.write_json(common.with_suffix('.proof.json'), {'source_sha256': binding['source_checkpoint_sha256'],
                          'modified_checkpoint_sha256': engine.sha256(common), 'target': target,
                          'model_optimizer_RNG_camera_state_tensor_identical': True,
                          'changed_only': ['opt.iterations', 'testing_iterations', 'saving_iterations']})
    checkpoint = directory / 'checkpoints/resume.pt'
    checkpoint.parent.mkdir(exist_ok=True)
    shutil.copyfile(common, checkpoint)
    digest = engine.sha256(checkpoint)
    manifest = engine.read_json(SOURCE / 'checkpoints/resume.json')
    manifest.update(path=str(checkpoint), sha256=digest, size_bytes=checkpoint.stat().st_size)
    engine.write_json(checkpoint.with_suffix('.json'), manifest)
    side = engine.read_json(SOURCE / 'checkpoints/resume.metrics.json')
    side.update(config=config, checkpoint_sha256=digest, association_data_sha256=binding['sha256'])
    engine.write_json(checkpoint.with_suffix('.metrics.json'), side)
    engine.write_json(directory / 'branch.json', {'source': str(SOURCE), 'destination': str(directory), 'iteration': 50000,
                      'identical_full_checkpoint_sha256': digest, 'source_checkpoint_sha256': binding['source_checkpoint_sha256'],
                      'tensor_optimizer_RNG_camera_state_unchanged': True, 'expanded_horizon_only': target + 1})
    engine.write_json(directory / 'reproduction.json', {'id': name, 'commands': [], 'status': 'initialized',
                      'target_iteration': target, 'native_width': 1600, 'matched_full_state': True})
    return directory


def train(directory, repo, target, deadline, stop=None):
    command = [*engine.command_prefix(repo), 'train_pidg.py', '-s', str(DATA), '-m', str(directory),
               '--conf', str(directory / 'model_config.py'), '--load2gpu_on_the_fly', '--resolution', '1600',
               '--save_image_dir', str(directory / 'diagnostics'), '--iterations', str(target + 1),
               '--checkpoint_interval', '500', '--resume_checkpoint', str(directory / 'checkpoints/resume.pt')]
    env = engine.env_for(directory)
    env['PIDG_TUNING_STOP_AFTER'] = str(stop or target)
    return engine.execute(command, directory, 'smoke' if target < 50100 else 'long', env, deadline)


def progress():
    state = engine.read_json(RUNS / 'controller.json') if (RUNS / 'controller.json').exists() else {}
    rows = []
    for name in NAMES:
        directory = RUNS / 'pilot' / name / 'rubber_duck'
        receipt = directory / 'checkpoints/resume.json'
        records = directory / 'association_loss.jsonl'
        losses = [engine.json.loads(s) for s in records.read_text().splitlines()] if records.exists() else []
        rows.append({'name': name, 'iteration': engine.read_json(receipt)['iteration'] if receipt.exists() else 50000,
                     'last_association_loss': losses[-1] if losses else None})
    engine.write_json(REPORT / 'progress.json', {'controller': state, 'updated_utc': engine.now(), 'models': rows})


def preflight(repo, binding, deadline):
    proof = REPORT / 'preflight.json'
    if proof.exists():
        saved = engine.read_json(proof)
        assert saved['status'] == 'passed' and saved['binding_sha256'] == binding['sha256']
        if 'core_sha256' in saved:
            assert saved['core_sha256'] == core_hashes(), 'Preflight training core changed'
        return
    environment = engine.env_for(RUNS)
    environment['PIDG_COLOR_REPAIR_GPU_TEST'] = '1'
    test = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_association_training.py',
                           'tests/test_object_membership.py', 'tests/test_color_repair_gpu.py', '-q', '-p', 'no:cacheprovider',
                           '--basetemp=tmp/association_preflight_v1'], cwd=ROOT, env=environment, capture_output=True, text=True)
    engine.write_text_atomic(REPORT / 'tests_gpu.log', test.stdout + test.stderr)
    assert test.returncode == 0, 'GPU/CPU preflight tests failed'
    probe = subprocess.run([sys.executable, '-u', '-m', 'variants.association_training.probe'], cwd=ROOT,
                           env=environment, capture_output=True, text=True)
    engine.write_text_atomic(REPORT / 'optimizer_probe.log', probe.stdout + probe.stderr)
    assert probe.returncode == 0, 'Actual isolated association update failed'
    off = branch(NAMES[0], 50005, binding, True)
    assert train(off, repo, 50005, deadline) == 75
    on = branch(NAMES[1], 50040, binding, True)
    assert train(on, repo, 50040, deadline, stop=50020) == 75
    assert train(on, repo, 50040, deadline) == 75
    log = (on / 'train.log').read_text()
    assert 'FULL_CHECKPOINT_RESTORED completed=50020 next=50021' in log
    assert len((on / 'association_loss.jsonl').read_text().splitlines()) == 2
    assert not (off / 'association_loss.jsonl').exists()
    for d, iteration in [(off, 50005), (on, 50040)]:
        manifest = engine.read_json(d / 'checkpoints/resume.json')
        assert manifest['iteration'] == iteration and manifest['sha256'] == engine.sha256(d / 'checkpoints/resume.pt')
        s = torch.load(d / 'checkpoints/resume.pt', map_location='cpu', weights_only=False)
        assert len(s['gaussian_parameters']['_xyz']) == binding['point_count']
    engine.write_json(proof, {'status': 'passed', 'binding_sha256': binding['sha256'], 'native_width': 1600,
                      'flags_off_smoke_updates': 5, 'on_smoke_updates': 40, 'resume_boundary': [50020, 50040],
                      'actual_new_loss_optimizer_update': True, 'zero_unknown_ID_fill': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--hours', type=float, default=4.)
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    repo = prepare()
    binding = prepare_data()
    engine.RUNS, engine.REPORTS, engine.repo_global = RUNS, REPORT, repo
    deadline = datetime.now(timezone.utc) + timedelta(hours=args.hours)
    state = {'status': 'preflight', 'pid': os.getpid(), 'started_utc': engine.now(), 'deadline_utc': deadline.isoformat()}
    event = threading.Event()
    def watch():
        while not event.wait(5):
            progress()
    with engine.lock():
        engine.write_json(RUNS / 'controller.json', state)
        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        try:
            preflight(repo, binding, deadline)
            if args.preflight_only:
                state['status'] = 'preflight_passed'
                return
            frozen = freeze()
            proof = engine.read_json(REPORT / 'preflight.json')
            proof.setdefault('core_sha256', frozen['core_sha256'])
            engine.write_json(REPORT / 'preflight.json', proof)
            for name in NAMES:
                branch(name, 52000, binding)
            a, b = [RUNS / 'pilot' / name / 'rubber_duck' for name in NAMES]
            if all(engine.read_json(d / 'checkpoints/resume.json')['iteration'] == 50000 for d in [a, b]):
                assert engine.sha256(a / 'checkpoints/resume.pt') == engine.sha256(b / 'checkpoints/resume.pt')
            ca, cb = [engine.read_json(d / 'metric_config.json') for d in [a, b]]
            assert {k for k in ca if ca[k] != cb[k]} == {'association_training_enabled'}
            state['status'] = 'training'
            engine.write_json(RUNS / 'controller.json', state)
            for directory in [a, b]:
                if engine.read_json(directory / 'checkpoints/resume.json')['iteration'] < 52000:
                    assert train(directory, repo, 52000, deadline) == 75
                assert engine.read_json(directory / 'checkpoints/resume.json')['iteration'] == 52000
            sampled = [[engine.json.loads(s) for s in (d / 'camera_sampling.jsonl').read_text().splitlines()] for d in [a, b]]
            assert sampled[0] == sampled[1] and len(sampled[0]) == 2000
            assert engine.sha256(SOURCE / 'checkpoints/resume.pt') == binding['source_checkpoint_sha256']
            freeze()
            state['status'] = 'training_completed_pending_evaluation'
        except Exception as exc:
            state.update(status='failed', error=str(exc))
            raise
        finally:
            event.set()
            thread.join(timeout=6)
            state['finished_utc'] = engine.now()
            engine.write_json(RUNS / 'controller.json', state)
            progress()


if __name__ == '__main__':
    main()
