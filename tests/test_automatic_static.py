import numpy as np
import pytest
from variants.staged_reconstruction.automatic_static.geometry import evidence, region_support


def observations(moving=False, baseline=1.):
    ids = np.arange(1, 65, 4)
    intrinsic = np.array([[400., 0, 200], [0, 400., 112], [0, 0, 1.]])
    views, uv = {}, []
    for i, n in enumerate(ids):
        view = np.eye(4)
        view[0, 3] = -baseline*i/(len(ids)-1)
        views[int(n)] = view
        xyz = np.array([.4*np.sin(i*.7) if moving else 0., .1, 4., 1.])
        q = intrinsic@(view@xyz)[:3]
        uv.append(q[:2]/q[2])
    return ids, np.asarray(uv), views, intrinsic


def test_stationary_world_point_accepted_despite_camera_induced_pixel_motion():
    args = observations()
    assert np.linalg.norm(args[1][-1]-args[1][0]) > 50
    result = evidence(*args)
    assert result['label'] == 'static_candidate'
    assert result['held_p90_px'] < 1e-6
    assert np.allclose(result['xyz'], [0, .1, 4])


def test_moving_point_is_incompatible_with_fixed_world_point():
    assert evidence(*observations(moving=True))['label'] == 'incompatible_fixed_point'


def test_low_parallax_is_unknown_instead_of_static():
    assert evidence(*observations(baseline=.0001))['label'] == 'unknown'


def test_short_interval_is_unknown():
    ids, uv, views, k = observations()
    assert evidence(ids[:6], uv[:6], views, k)['label'] == 'unknown'


def test_held_observations_do_not_fit_themselves():
    ids, uv, views, k = observations()
    uv[1::2, 1] += 20
    result = evidence(ids, uv, views, k)
    assert result['label'] == 'incompatible_fixed_point'
    assert result['held_median_px'] > 19


def test_regions_require_multiple_geometric_seeds_and_reject_conflict():
    labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3]])
    static = [[0, 0], [1, 1], [2, 0], [3, 1], [0, 2]]
    mask = region_support(labels, static, [[3, 0]])
    assert mask[0, 0] and not mask[0, 2] and not mask[2, 0] and not mask[2, 2]


def test_no_static_seeds_means_unknown_regardless_of_image_region():
    assert not region_support(np.zeros((3, 3), int), [], []).any()


def test_disabled_discovery_does_not_read_data_or_create_masks():
    import subprocess, sys
    result = subprocess.run([sys.executable, '-m', 'variants.staged_reconstruction.automatic_static.discover'],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'opt-in' in result.stderr
    assert 'AUTOMATIC_STATIC_DISCOVERY' not in result.stdout


def test_unknown_pixels_have_no_direct_training_gradient():
    import torch
    from variants.staged_reconstruction.automatic_static.geometry import static_objective
    image = torch.full((3, 3, 3), .5, requires_grad=True)
    alpha = torch.full((1, 3, 3), .5, requires_grad=True)
    mask = torch.zeros((3, 3), dtype=torch.bool); mask[0, 0] = True
    loss = static_objective(image, alpha, torch.zeros_like(image), mask)
    gi, ga = torch.autograd.grad(loss, [image, alpha])
    assert gi[:, 0, 0].abs().sum() > 0 and ga[0, 0, 0] < 0
    assert gi[:, 1, 1].abs().sum() == 0 and ga[0, 1, 1] == 0


def test_no_inferred_support_does_not_become_empty_scene_target():
    import torch
    from variants.staged_reconstruction.automatic_static.geometry import static_objective
    with pytest.raises(ValueError):
        static_objective(torch.zeros(3, 2, 2), torch.zeros(1, 2, 2), torch.zeros(3, 2, 2), torch.zeros(2, 2, dtype=torch.bool))


def test_disabled_training_does_not_initialize_renderer():
    import subprocess, sys
    result = subprocess.run([sys.executable, '-m', 'variants.staged_reconstruction.automatic_static.train'], capture_output=True, text=True)
    assert result.returncode != 0 and 'opt-in' in result.stderr
    assert 'ninja' not in result.stdout
