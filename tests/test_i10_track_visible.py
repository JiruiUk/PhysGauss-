import numpy as np
from variants.staged_reconstruction.i10_track_neighborhood.visible_core import footprint_ids, choose, reciprocal_result


def inputs():
    xyz = np.zeros((12, 3, 3)); xyz[..., 2] = 4.; xyz[:, 1, 0] = .1; xyz[:, 2, 0] = .2
    projected = np.zeros((12, 3, 2)); projected[:, 1, 0] = .8; projected[:, 2, 0] = 1.5
    obs = {i:[dict(passed=True, known=True) for _ in range(12)] for i in range(3)}
    return xyz, projected, obs


def select(projected, xyz, obs):
    return choose(np.zeros((12, 2)), xyz, projected, xyz[..., 2], [0,1,2], obs, list(range(0,12,2)), list(range(1,12,2)), {})


def test_hidden_nearest_projection_is_not_chosen():
    xyz, projected, obs = inputs(); obs[0] = [dict(passed=False) for _ in range(12)]
    result = select(projected, xyz, obs)
    assert result['accepted'] and result['row'] == 1


def test_held_frames_validate_without_choosing_fit_id():
    xyz, projected, obs = inputs(); projected[1::2, 0, 0] = 3.
    result = select(projected, xyz, obs)
    assert result['row'] == 0 and not result['accepted']


def test_tied_distant_visible_surfaces_remain_ambiguous():
    xyz, projected, obs = inputs(); projected[:, 1] = projected[:, 0]
    result = select(projected, xyz, obs)
    assert result['row'] == 0 and not result['checks']['alternative']


def test_empty_and_behind_camera_stay_unknown():
    xyz, projected, obs = inputs(); xyz[..., 2] = -1.
    assert 'row' not in select(projected, xyz, obs)


def test_native_joint_id_footprint_excludes_static_and_invalid():
    ids = np.full((20, 12, 12), -1); ids[0, 5, 5] = 2; ids[1, 5, 5] = 5; ids[2, 5, 5] = 6
    assert footprint_ids(ids, [5.4,5.3], 5, 10) == [0,1]


def test_held_visibility_cannot_rescue_invisible_fit():
    xyz, projected, obs = inputs()
    for i in obs:
        for t in range(0,12,2):obs[i][t]['passed'] = False
    assert 'row' not in select(projected, xyz, obs)


def test_reciprocal_missing_or_distant_stays_unknown():
    xyz, projected, _ = inputs(); p=dict(equivalent_max_world_distance=.01,equivalent_max_projection_distance_px=.75)
    assert not reciprocal_result({}, {}, xyz, projected, p)['passed']
    assert not reciprocal_result(dict(row=0,accepted=True),dict(row=1,accepted=True),xyz,projected,p)['passed']
