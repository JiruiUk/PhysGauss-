"""CPU-only tests for dataset exclusion and strict dependent-queue gating."""
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'variants/random_frame_mask'
spec = importlib.util.spec_from_file_location('frame_mask_runner', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from prepare import split_frames


def documents():
    train = {'camera_angle_x': 0.7, 'frames': [{'file_path': f'train/{i:04d}', 'time': i / 149,
                                               'transform_matrix': [[i]]} for i in range(150)]}
    test = {'camera_angle_x': 0.7, 'frames': [{'file_path': f'test/{i:04d}', 'time': i / 49} for i in range(50)]}
    return train, test


def test_nested_deterministic_whole_frame_dropout_preserves_camera_data():
    train, test = documents()
    saved = json.dumps([train, test], sort_keys=True)
    state = random.getstate()
    outputs = split_frames(train, test)
    assert random.getstate() == state
    assert outputs == split_frames(train, test)
    previous = set()
    for row, keep, removed in zip(outputs, [135, 105, 75, 45, 15], [15, 45, 75, 105, 135]):
        kept, dropped = set(row['retained_ids']), set(row['dropped_ids'])
        assert len(kept) == keep and len(dropped) == removed
        assert not kept & dropped
        assert previous <= dropped
        assert kept | dropped == {f['file_path'] for f in train['frames']}
        assert row['test'] == test
        assert row['train']['frames'] == [f for f in train['frames'] if f['file_path'] in kept]
        assert row['heldout']['frames'] == [f for f in train['frames'] if f['file_path'] in dropped]
        previous = dropped
    assert json.dumps([train, test], sort_keys=True) == saved
    assert split_frames(train, test, seed=1)[0]['dropped_ids'] != outputs[0]['dropped_ids']


@pytest.mark.parametrize('percentage', [-1, 0, 100, 101, 10.0, True])
def test_invalid_dropout_is_rejected(percentage):
    with pytest.raises(ValueError):
        split_frames(*documents(), percentages=[percentage])


def test_original_overlap_cannot_be_hidden():
    train, test = documents()
    test['frames'][0]['file_path'] = train['frames'][0]['file_path']
    with pytest.raises(ValueError, match='overlap'):
        split_frames(train, test)


def fake_finished_run(path, steps=True):
    metric = {'PSNR': 27.0, 'SSIM': .97, 'LPIPS': .03, 'MS-SSIM': .97}
    info = {'status': 'completed', 'iterations': 50000, 'resolution_width': 1600, 'seed': 0,
            'metrics': {'ours_best': metric}, 'completed_steps': ['train', 'render', 'metrics'] if steps else ['train']}
    runner.write_json(path / 'reproduction.json', info)
    runner.write_json(path / 'results.json', {'ours_best': metric})
    for kind in ('gt', 'renders'):
        directory = path / 'test/ours_best' / kind
        directory.mkdir(parents=True)
        for i in range(50):
            (directory / f'{i:05d}.png').write_bytes(b'test fixture')


def test_dependency_requires_controller_exit_metrics_and_every_render(tmp_path, monkeypatch):
    mlp, baseline = tmp_path / 'mlp', tmp_path / 'baseline'
    monkeypatch.setattr(runner, 'alive', lambda pid: False)
    runner.write_json(mlp / 'controller.json', {'status': 'running', 'pid': 123})
    assert not runner.dependency_ready(mlp, baseline)[0]
    fake_finished_run(baseline)
    for name in ('W512_D0', 'W256_D2', 'W512_D2'):
        fake_finished_run(mlp / ('long_' + name) / 'rubber_duck')
    for state in ('running', 'failed', 'saved_stopped'):
        runner.write_json(mlp / 'controller.json', {'status': state, 'pid': 123})
        assert not runner.dependency_ready(mlp, baseline)[0]
    runner.write_json(mlp / 'controller.json', {'status': 'completed', 'pid': 123})
    assert runner.dependency_ready(mlp, baseline)[0]
    monkeypatch.setattr(runner, 'alive', lambda pid: pid == 123)
    assert not runner.dependency_ready(mlp, baseline)[0]
    monkeypatch.setattr(runner, 'alive', lambda pid: False)
    victim = mlp / 'long_W512_D2/rubber_duck/test/ours_best/renders/00049.png'
    victim.unlink()
    assert not runner.dependency_ready(mlp, baseline)[0]
    victim.write_bytes(b'test fixture')
    (mlp / 'stop_queue.request').write_text('pause')
    assert not runner.dependency_ready(mlp, baseline)[0]


def test_heldout_commands_use_distinct_source_and_output_paths(tmp_path):
    directory = tmp_path / 'run'
    dataset, heldout = tmp_path / 'data', tmp_path / 'heldout_data'
    commands = dict(runner.make_commands(directory, dataset, heldout))
    assert commands['train'][commands['train'].index('-s') + 1] == str(dataset)
    assert commands['render'][commands['render'].index('-m') + 1] == str(directory)
    assert commands['heldout_render'][commands['heldout_render'].index('-m') + 1] == str(directory / 'masked_holdout')
    assert commands['heldout_render'][commands['heldout_render'].index('-s') + 1] == str(heldout)
    assert 'train_pidg.py' not in commands['heldout_render']
    assert '--resume_checkpoint' not in commands['train']
    assert commands['train'][commands['train'].index('--iterations') + 1] == '50000'


def test_waiting_watcher_does_not_import_torch_or_claim_a_gpu():
    code = f"import runpy,sys;runpy.run_path({str(HERE / 'run.py')!r},run_name='cpu_probe');assert 'torch' not in sys.modules;print('ok')"
    assert subprocess.check_output([sys.executable, '-c', code], cwd=ROOT, text=True).strip() == 'ok'


def test_source_freeze_rejects_changes_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'REPORTS', tmp_path)
    frozen = {'baseline_runtime': {'train_pidg.py': 'original'}}
    runner.write_json(tmp_path / 'source_freeze.json', frozen)
    monkeypatch.setattr(runner, 'current_freeze', lambda: frozen)
    runner.check_freeze()
    monkeypatch.setattr(runner, 'current_freeze', lambda: {'baseline_runtime': {'train_pidg.py': 'changed'}})
    with pytest.raises(RuntimeError, match='Frozen baseline'):
        runner.check_freeze()
