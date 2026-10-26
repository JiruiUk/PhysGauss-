"""A held-out clock must not leak images or previously fitted weights."""
import json
from pathlib import Path
import pytest

from variants.kinematic_resplit.data import partition, audit_original, materialize
from variants.kinematic_resplit.run import branch_config


def frames(ids):
    return [{'file_path': f'train/{i:04d}', 'time': 0., 'transform_matrix': []} for i in ids]


def test_three_way_partition_keeps_official_test_and_global_clock():
    parts = partition(frames(range(1, 201)))
    ids = {s: {int(Path(f['file_path']).stem) for f in fs} for s, fs in parts.items()}
    assert {s: len(v) for s, v in ids.items()} == {'train': 120, 'val': 30, 'test': 50}
    assert ids['test'] == set(range(1, 201, 4))
    assert ids['train'] | ids['val'] | ids['test'] == set(range(1, 201))
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not ids[a] & ids[b]
        assert not {f['time'] for f in parts[a]} & {f['time'] for f in parts[b]}
    for fs in parts.values():
        assert all(f['time'] == (int(Path(f['file_path']).stem)-1)/199 for f in fs)


def test_partition_rejects_duplicates_or_missing_acquisitions():
    for fs in [frames(range(1, 200)), frames(range(1, 200))+frames([1])]:
        with pytest.raises(ValueError): partition(fs)


def test_old_validation_is_test_not_an_independent_split(tmp_path):
    for split, ids in [('train', range(2, 5)), ('test', [1])]:
        (tmp_path/f'transforms_{split}.json').write_text(json.dumps({'frames': frames(ids)}))
    audit = audit_original(tmp_path)
    assert audit['independent_validation_exists'] is False
    assert audit['validation_test_role_overlap'] == 1
    assert audit['train_test_image_overlap'] == []


def test_train_view_contains_no_test_rgb(tmp_path):
    source, destination = tmp_path/'source', tmp_path/'derived'
    (source/'train').mkdir(parents=True)
    for i in range(1, 201): (source/'train'/f'{i:04d}.png').write_bytes(b'RGB placeholder')
    parts = partition(frames(range(1, 201)))
    materialize(source, destination, {'camera_angle_x': .5}, parts, point_count=4)
    training = destination/'training'
    assert len(list((training/'train').glob('*.png'))) == 150
    assert not (training/'train/0001.png').exists()
    assert len(json.loads((training/'transforms_test.json').read_text())['frames']) == 30
    assert (destination/'final_test/train/0001.png').exists()
    assert not (training/'flows_flo').exists()
    assert not (training/'depth_distill').exists()


def test_full_state_branch_only_allows_declared_loss_weight_change():
    original = {'enabled': True, 'kinematic_weight': 0., 'kinematic_start': 1000}
    branch = dict(original, kinematic_weight=.001)
    branch_config(original, branch)
    with pytest.raises(ValueError): branch_config(original, dict(branch, pixel_loss='mse'))


def test_final_test_cannot_run_before_both_selections_are_locked(tmp_path, monkeypatch):
    from variants.kinematic_resplit import run
    monkeypatch.setattr(run, 'RUNS', tmp_path)
    with pytest.raises(ValueError, match='still locked'):
        run.render(tmp_path/'model', None, None, split='test')


def test_disabled_new_runtime_keeps_legacy_time_and_does_not_access_scene(tmp_path):
    import torch
    from variants.kinematic_resplit.runtime import Tuning
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'enabled': False, 'kinematic_weight': .001}))
    runtime = Tuning(path, tmp_path)
    runtime.configure_scene(None, 'unused')
    before = torch.get_rng_state().clone()
    assert runtime.kinematic_loss(10, None, None, None) is None
    assert torch.equal(before, torch.get_rng_state())
    fid = torch.tensor([.5])
    assert torch.equal(runtime.physics_time(fid, 120), fid/120)


def test_branch_relocation_preserves_all_optimizer_options(tmp_path):
    from variants.kinematic_resplit.checkpoint import validate_relocation
    a, b = tmp_path/'V3', tmp_path/'V4'
    saved = {'save_image_dir': str(a/'diagnostics'), 'position_lr_init': .001}
    current = dict(saved, save_image_dir=str(b/'diagnostics'))
    receipt = {'source': str(a), 'destination': str(b), 'identical_full_checkpoint_sha256': 'same'}
    assert validate_relocation(saved, current, receipt, 'same')
    with pytest.raises(ValueError): validate_relocation(saved, dict(current, position_lr_init=.01), receipt, 'same')
    with pytest.raises(ValueError): validate_relocation(saved, current, receipt, 'changed')
    with pytest.raises(ValueError): validate_relocation(saved, current, None, 'same')


def test_metrics_require_complete_finite_per_frame_results():
    from variants.kinematic_resplit.run import verify_metrics, engine
    results = {'ours_best': {k: .5 for k in engine.METRICS}}
    views = {'ours_best': {k: {'00000.png': .4, '00001.png': .6} for k in engine.METRICS}}
    verify_metrics(results, views, ['best'], 2)
    with pytest.raises(ValueError): verify_metrics(results, views, ['best'], 3)
    views['ours_best']['PSNR']['00000.png'] = float('nan')
    with pytest.raises(ValueError): verify_metrics(results, views, ['best'], 2)
