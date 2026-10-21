"""Verify split coverage and per-image metric aggregation before full evaluation."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mask_split_evaluator', ROOT / 'variants/random_frame_mask/evaluate_splits.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def frame(name, time=0):
    return {'file_path': name, 'time': time, 'transform_matrix': [[1]]}


def test_split_requires_disjoint_complete_unchanged_training_views():
    train = [frame('train/0'), frame('train/1', 1)]
    test = [frame('test/0')]
    evaluation.validate_partition(train, test, train[:1], train[1:])
    evaluation.validate_partition(train, test, train, [])
    with pytest.raises(ValueError, match='overlap'):
        evaluation.validate_partition(train, test, train, train[1:])
    with pytest.raises(ValueError, match='coverage'):
        evaluation.validate_partition(train, test, train[:1], [])
    with pytest.raises(ValueError, match='metadata'):
        evaluation.validate_partition(train, test, train[:1], [frame('train/1', .5)])
    with pytest.raises(ValueError, match='Duplicate'):
        evaluation.validate_partition(train, test, train + train[:1], [])


def test_union_mean_weights_frames_not_subsets_and_missing_is_na():
    def row(name, value):
        return {'source_id': name, 'metrics': {k: value for k in evaluation.METRICS}}
    kept = [row('a', 40), row('b', 40), row('c', 40)]
    dropped = [row('d', 20)]
    assert evaluation.mean_metrics(kept + dropped)['PSNR'] == 35
    assert evaluation.mean_metrics([]) is None
    with pytest.raises(ValueError, match='Duplicate'):
        evaluation.mean_metrics(kept + kept)
    with pytest.raises(ValueError, match='Nonfinite'):
        evaluation.mean_metrics([row('e', float('nan'))])


def test_cached_metrics_must_cover_all_expected_images(tmp_path):
    raw = tmp_path / 'test/ours_best'
    for kind in ('renders', 'gt'):
        (raw / kind).mkdir(parents=True)
        (raw / kind / '00000.png').touch()
    evaluation.validate_images(raw, 1)
    with pytest.raises(ValueError, match='Incomplete'):
        evaluation.validate_images(raw, 2)
    (raw / 'gt/00001.png').touch()
    with pytest.raises(ValueError, match='extra'):
        evaluation.validate_images(raw, 1)


def test_retained_render_alias_changes_only_loader_path_not_time_or_pose():
    original = [frame('train/0002', .3), frame('train/0020', .8)]
    actual = evaluation.retained_test_frames(original)
    assert [f['file_path'] for f in actual] == ['retained_eval_images/0002', 'retained_eval_images/0020']
    for before, after in zip(original, actual):
        assert {k:v for k,v in before.items() if k != 'file_path'} == {k:v for k,v in after.items() if k != 'file_path'}
    assert original[0]['file_path'] == 'train/0002'
