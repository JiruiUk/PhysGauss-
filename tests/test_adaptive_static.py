import numpy as np
import pytest
import torch
from variants.staged_reconstruction.adaptive_static.geometry import (
    backproject, camera_center, depth_consistency, fuse_voxels, patch_features, sweep, fit_plane, plane_intersections)
from variants.staged_reconstruction.adaptive_static.model import AdaptiveGaussian, append_points, prune_points
from variants.staged_reconstruction.train_static import optimizer_for


def synthetic_plane():
    h, w = 48, 64
    k = torch.tensor([[80., 0, 32], [0, 80., 24], [0, 0, 1.]])
    views = torch.eye(4).repeat(7, 1, 1)
    views[:, 0, 3] = torch.tensor([0., -.25, .25, -.5, .5, -.75, .75])
    images = []
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    for view in views:
        x = (xx-32)/80*4-view[0, 3]; y = (yy-24)/80*4
        gray = .5+.12*torch.sin(11*x+5*y)+.12*torch.sin(17*y-7*x)+.1*torch.cos(23*x+19*y)
        images.append(torch.stack([gray, gray*.8, gray*.6]))
    return torch.stack(images), views, k


def test_dense_plane_recovers_independent_known_depth():
    images, views, k = synthetic_plane()
    features, valid = patch_features(images)
    uv = torch.tensor([[25., 19.], [36, 20], [30, 29], [40, 30]])
    depths = torch.linspace(3., 5., 161)
    _, depth, accepted, _, _, _ = sweep(uv, features, valid, views, k, depths, 0, [1, 2, 3, 4, 5, 6], margin=.001, cost_limit=.15)
    assert accepted.all()
    assert (depth-4).abs().max() < .06


def test_held_images_can_reject_a_fit_not_explainable_by_one_static_surface():
    images, views, k = synthetic_plane()
    images[5:] = 1-images[5:]
    features, valid = patch_features(images)
    uv = torch.tensor([[25., 19.], [36, 20], [30, 29], [40, 30]])
    _, _, accepted, _, _, _ = sweep(uv, features, valid, views, k, torch.linspace(3., 5., 161), 0,
                                  [1, 2, 3, 4, 5, 6], margin=.001, cost_limit=.15)
    assert not accepted.any()


def test_uniform_texture_does_not_create_a_unique_geometric_depth():
    features, valid = patch_features(torch.ones(7, 3, 48, 64)*.5)
    assert not valid[:, 5:-5, 5:-5].any()
    views = torch.eye(4).repeat(7, 1, 1); views[:, 0, 3] = torch.arange(7)*.1
    _, _, accepted, _, _, _ = sweep(torch.tensor([[25., 19.]]), features, valid, views,
        torch.tensor([[80., 0, 32], [0, 80, 24], [0, 0, 1.]]), torch.linspace(3., 5., 20), 0, [1, 2, 3, 4, 5, 6])
    assert not accepted.any()


def test_identical_grayscale_edges_with_different_chroma_are_not_matches():
    images, views, k = synthetic_plane()
    gray = images[:, 0:1]
    images[1:] = torch.cat([gray[1:]*.05, gray[1:]*.15, gray[1:]], 1)
    images[0] = torch.cat([gray[0], gray[0]*.9, gray[0]*.05], 0)
    features, valid = patch_features(images)
    _, _, accepted, _, _, _ = sweep(torch.tensor([[25., 19.]]), features, valid, views, k,
        torch.linspace(3., 5., 161), 0, [1, 2, 3, 4, 5, 6], margin=.001, cost_limit=.15)
    assert not accepted.any()


def test_missing_source_depth_remains_unknown_and_wrong_depth_is_rejected():
    xyz = np.array([[0., 0, 4]])
    uv = np.array([[32., 24.]])
    k = np.array([[80., 0, 32], [0, 80, 24], [0, 0, 1.]])
    view = np.eye(4)
    assert not depth_consistency(xyz, uv, np.zeros((48, 64)), view, view, k)[0].any()
    assert not depth_consistency(xyz, uv, np.ones((48, 64))*2, view, view, k)[0].any()
    assert depth_consistency(xyz, uv, np.ones((48, 64))*4, view, view, k)[0].all()


def test_voxel_fusion_preserves_multiple_source_views():
    xyz = np.array([[.001, 0, 4], [.002, 0, 4], [.1, 0, 4]])
    support = np.eye(3, dtype=bool)
    x, c, s = fuse_voxels(xyz, np.ones((3, 3)), support, .025)
    assert len(x) == 2 and s[0].sum() == 2 and np.allclose(c, 1)


def test_verified_plane_interpolates_untextured_interior_with_explicit_prior():
    x, y = np.meshgrid(np.linspace(-.5, .5, 6), np.linspace(-.5, .5, 6))
    points = np.c_[x.ravel(), y.ravel(), np.full(x.size, 4.)]
    plane = fit_plane(points); assert plane is not None
    xyz, depth, valid = plane_intersections(np.array([[32., 24.]]), plane, np.eye(4),
                                           np.array([[80., 0, 32], [0, 80, 24], [0, 0, 1.]]))
    assert valid.all() and np.allclose(xyz, [[0, 0, 4]]) and np.allclose(depth, 4)


def test_collinear_anchors_cannot_determine_a_surface_plane():
    assert fit_plane(np.c_[np.linspace(-1, 1, 20), np.zeros(20), np.ones(20)*4]) is None


def test_plane_held_points_cannot_fit_themselves():
    x, y = np.meshgrid(np.linspace(-.5, .5, 6), np.linspace(-.5, .5, 6))
    points = np.c_[x.ravel(), y.ravel(), np.full(x.size, 4.)]; points[1::2, 2] += .2
    assert fit_plane(points) is None


def make_model(tmp_path):
    path = tmp_path/'initial.npz'
    np.savez(path, xyz=np.array([[0., 0, 0], [.1, 0, 0]], np.float32),
             rgb=np.ones((2, 3), np.float32)*.5, spacing=np.asarray(.04), point_ids=np.array([0, 1]))
    model = AdaptiveGaussian(path, device='cpu'); opt = optimizer_for(model)
    sum(p.square().sum() for p in model.parameters()).backward(); opt.step(); opt.zero_grad()
    return path, model, opt


def test_birth_preserves_existing_parameters_and_adam_moments(tmp_path):
    _, model, opt = make_model(tmp_path)
    values = {n: p.detach().clone() for n, p in model.named_parameters()}
    states = {g['name']: {k: v.clone() if torch.is_tensor(v) else v for k, v in opt.state[g['params'][0]].items()} for g in opt.param_groups}
    append_points(model, opt, [[1., 0, 0]], [[.2, .4, .6]], [2], 3000)
    assert model.point_ids.tolist() == [0, 1, 2] and model.born_at.tolist() == [0, 0, 3000]
    for group in opt.param_groups:
        name, p = group['name'], group['params'][0]
        assert torch.equal(p[:2], values[name])
        state = opt.state[p]
        assert state['step'] == states[name]['step']
        for key in ['exp_avg', 'exp_avg_sq']:
            assert torch.equal(state[key][:2], states[name][key]) and not state[key][2].any()


def test_new_rows_get_gradients_and_survive_exact_topology_restore(tmp_path):
    path, model, opt = make_model(tmp_path)
    append_points(model, opt, [[1., 0, 0]], [[.2, .4, .6]], [2], 30)
    loss = sum(p.square().sum() for p in model.parameters()); loss.backward()
    assert all(p.grad[-1].abs().sum() > 0 for n, p in model.named_parameters() if n != '_features_rest')
    opt.step(); opt.zero_grad()
    restored = AdaptiveGaussian(path, device='cpu'); restored.load_payload(model.state_dict()); opt2 = optimizer_for(restored)
    opt2.load_state_dict(opt.state_dict())
    assert all(torch.equal(v, restored.state_dict()[n]) for n, v in model.state_dict().items())
    assert [int(opt2.state[g['params'][0]]['step']) for g in opt2.param_groups] == [2]*6


def test_pruning_only_removes_trial_new_rows_and_keeps_old_adam(tmp_path):
    _, model, opt = make_model(tmp_path)
    append_points(model, opt, [[1., 0, 0]], [[.2, .4, .6]], [2], 30, opacity=.01)
    removed = prune_points(model, opt, 2000)
    assert removed == [2] and model.point_ids.tolist() == [0, 1]
    assert all(int(opt.state[g['params'][0]]['step']) == 1 for g in opt.param_groups)


def test_duplicate_point_birth_is_rejected(tmp_path):
    _, model, opt = make_model(tmp_path)
    with pytest.raises(AssertionError): append_points(model, opt, [[1., 0, 0]], [[.2, .4, .6]], [1], 30)


def test_residual_alone_cannot_birth_without_geometry_or_duplicate_ever_id():
    from variants.staged_reconstruction.adaptive_static.topology import select_births
    xyz = np.array([[0., 0, 4], [1., 0, 4], [2., 0, 4], [3., 0, 4]])
    protocol = {'birth_min_support_views': 4, 'birth_min_distance': .03}
    assert select_births(xyz, xyz[:1], np.ones(4), np.array([5, 1, 4, 4]), {0, 2}, protocol, 512) == [3]
    assert select_births(xyz, xyz[:1], np.zeros(4), np.ones(4)*5, {0}, protocol, 512) == []
    assert select_births(xyz, xyz[:1], np.ones(4), np.ones(4)*5, {0}, protocol, 0) == []


@pytest.mark.parametrize('entry', ['prepare', 'train'])
def test_new_entries_are_disabled_by_default(entry):
    import subprocess, sys
    result = subprocess.run([sys.executable, '-m', 'variants.staged_reconstruction.adaptive_static.'+entry], capture_output=True, text=True)
    assert result.returncode != 0 and 'opt-in' in result.stderr
