from variants.staged_reconstruction.surface_trajectory.split import group_tracks, split_groups


CONFIG = dict(source_patch_diameter_px=4., duplicate_minimum_common_frames=3,
              duplicate_median_px=2., duplicate_maximum_px=3., weighted_support_overlap=.5)


def track(i, x):
    return dict(track_id=i, frames=list(range(8)), uv_400=[[x+j, 10.] for j in range(8)])


def test_duplicate_image_tracks_group_even_without_model_binding():
    groups, edges = group_tracks([track(0, 0), track(1, 1), track(2, 20)], [], CONFIG)
    assert groups == {0:[0, 1], 2:[2]} and len(edges) == 1


def test_large_shared_weight_keeps_disjoint_image_tracks_together():
    groups, _ = group_tracks([track(0, 0), track(1, 20)],
        [dict(track_id=i, rows=[7], weights=[1.]) for i in range(2)], CONFIG)
    assert groups == {0:[0, 1]}


def test_minimum_twenty_disjoint_holdout_groups_cannot_be_relaxed():
    result = split_groups({i:[i] for i in range(20)}, [dict(track_id=i) for i in range(20)])
    assert not result['admitted']


def test_group_count_priority_and_training_minimum_are_reproducible():
    groups = {i:list(range(i*10, i*10+(10 if i < 4 else 1))) for i in range(24)}
    bindings = [dict(track_id=t) for group in groups.values() for t in group]
    result = split_groups(groups, bindings)
    assert result == split_groups(groups, bindings)
    assert result['admitted'] and len(result['training_track_ids']) == 40
    assert len(result['holdout_groups']) == 20
    assert not set(result['training_track_ids'])&set(result['holdout_track_ids'])
