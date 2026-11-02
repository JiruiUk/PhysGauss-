import numpy as np
import pytest

from variants.object_membership.model import Config, optional_membership, reciprocal_mapping


def run(labels, frames=None):
    values = np.asarray(labels)
    frames = list(range(1, len(values) + 1)) if frames is None else frames
    return optional_membership(values, frames, Config(enabled=True))


def test_disabled_path_does_not_read_inputs():
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError('disabled input accessed')
    assert optional_membership(Poison(), Poison()) is None


def test_three_observations_required_before_causal_assignment():
    r = run([[0], [0], [0], [0]])
    assert r['historical_owner'][:, 0].tolist() == [-1, -1, 0, 0]


def test_visibility_loss_retains_history_without_claiming_current_support():
    r = run([[0], [0], [0], [-1], [-1], [0]])
    assert r['historical_owner'][:, 0].tolist() == [-1, -1, 0, 0, 0, 0]
    assert r['visible_owner'][:, 0].tolist() == [-1, -1, 0, -1, -1, 0]


def test_unknown_interrupts_qualification_streak():
    r = run([[0], [0], [-1], [0], [0], [0]])
    assert r['historical_owner'][:, 0].tolist() == [-1, -1, -1, -1, -1, 0]


def test_different_object_latches_conflict_even_after_smooth_reappearance():
    r = run([[0], [0], [0], [1], [-1], [0], [0], [0]])
    assert r['conflict'][:, 0].tolist() == [False, False, False, True, True, True, True, True]
    assert (r['historical_owner'][3:, 0] == -1).all()


def test_conflict_before_qualification_does_not_create_false_owner():
    assert (run([[0], [1], [1], [1]])['historical_owner'] == -1).all()


def test_all_unsupported_points_stay_unknown():
    assert (run([[-1, -1]] * 4)['historical_owner'] == -1).all()


def test_irregular_original_frames_count_observations_not_elapsed_frames():
    assert run([[2], [2], [2]], [2, 3, 6])['historical_owner'][-1, 0] == 2


def test_prefix_output_cannot_depend_on_future_contradiction():
    prefix = [[0], [0], [0]]
    assert np.array_equal(run(prefix)['historical_owner'], run(prefix + [[1]])['historical_owner'][:3])


def test_reciprocal_identity_accepts_permutations_not_numeric_label_equality():
    assert reciprocal_mapping([2, 0, 3, 1], [1, 3, 0, 2]).tolist() == [2, 0, 3, 1]


def test_reverse_disagreement_and_merge_do_not_bridge():
    assert reciprocal_mapping([0, 1, 2, 3], [1, 0, 2, 3]).tolist() == [-1, -1, 2, 3]
    assert reciprocal_mapping([0, 0, 2, 3], [0, -1, 2, 3]).tolist() == [-1, -1, 2, 3]


def test_noninteger_mapping_is_not_silently_truncated():
    with pytest.raises(ValueError):
        reciprocal_mapping([0.5, 1], [0, 1])


def test_domains_are_separate_and_local_same_number_is_not_a_global_id():
    a = run([[0], [0], [0]])
    b = run([[-1], [-1], [-1]])
    assert a['historical_owner'][-1, 0] == 0 and b['historical_owner'][-1, 0] == -1


@pytest.mark.parametrize('labels,frames', [([[0], [0]], [2, 2]), ([[0], [0]], [3, 2]), ([[4]], [2]), ([[0.5]], [2]), ([[-2]], [2]), ([0, 0], [2, 3])])
def test_invalid_evidence_fails_explicitly(labels, frames):
    with pytest.raises(ValueError):
        run(labels, frames)
