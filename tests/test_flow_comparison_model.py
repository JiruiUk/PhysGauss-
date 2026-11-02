import math
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from variants.flow_comparison.temporal import derivative
from variants.flow_comparison.local_momentum import fit_gradient, stress_tensor, residual
from variants.flow_comparison.flow import projected, sqrt_spd, pixel_flow

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_branch_options_reject_other_changes():
    from variants.flow_comparison.checkpoint import validate_options
    a = {'use_flow': False, 'save_image_dir': str(ROOT/'tmp/source/diagnostics'), 'lr': .1}
    b = dict(a, use_flow=True, save_image_dir=str(ROOT/'tmp/dest/diagnostics'))
    receipt = {'source': str(ROOT/'tmp/source'), 'destination': str(ROOT/'tmp/dest'), 'identical_full_checkpoint_sha256': 'abc'}
    assert validate_options(a, a, None, 'abc') is False
    assert validate_options(a, b, receipt, 'abc')
    with pytest.raises(ValueError): validate_options(a, dict(b, lr=.2), receipt, 'abc')
    with pytest.raises(ValueError): validate_options(a, b, receipt, 'changed')


def test_unused_head_and_disabled_module_preserve_parameters(tmp_path, monkeypatch):
    from variants.flow_comparison.model import configure_from_env
    spec = importlib.util.spec_from_file_location('flow_test_velocity', ROOT/'variants/monotonic_training/code/motion_utils/pidg.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    net = module.VelocityNet(max_particles=32)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    path = tmp_path/'config.json'; monkeypatch.setenv('PIDG_METRIC_CONFIG', str(path))
    path.write_text(json.dumps({'enabled': False, 'velocity_source': 'trajectory'}))
    configure_from_env(net)
    assert all(p.requires_grad for p in net.parameters())
    path.write_text(json.dumps({'enabled': True, 'velocity_source': 'trajectory', 'velocity_clock_mode': 'fixed_monotonic'}))
    configure_from_env(net)
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in before.items())
    assert not any(p.requires_grad for p in net.velocity_head.parameters())
    assert net.stress_head.weight.requires_grad


@pytest.mark.skipif(os.environ.get('PIDG_FLOW_GPU_TEST') != '1', reason='explicit CUDA raster check')
def test_cuda_raster_forward_matches_torch_projection(monkeypatch):
    import sys
    monkeypatch.delenv('PIDG_METRIC_CONFIG', raising=False)
    sys.path.insert(0, str(ROOT/'variants/flow_comparison/code'))
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.cameras import Camera
    from gaussian_renderer import render
    import numpy as np
    g = GaussianModel(3)
    g._xyz = torch.nn.Parameter(torch.randn(12, 3, device='cuda')*.1)
    g._scaling = torch.nn.Parameter(torch.randn(12, 3, device='cuda')*.1-3)
    q = torch.randn(12, 4, device='cuda'); q[:, 0] += 3
    g._rotation = torch.nn.Parameter(q)
    g._opacity = torch.nn.Parameter(torch.ones(12, 1, device='cuda')*2)
    camera = Camera(0, np.eye(3), np.array([0., 0., 3.]), math.pi/2, math.pi/3,
        torch.zeros(3, 225, 400), None, 'test', 0, fid=.5)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    warp = torch.zeros(12, 7, device='cuda'); warp[:, 0] = 1
    drot = torch.randn(12, 4, device='cuda')*.03
    dscale = torch.ones(12, 3, device='cuda')*.001
    package = render(camera, g, pipe, torch.zeros(3, device='cuda'), warp, drot, dscale, False,
        override_color=torch.zeros(12, 3, device='cuda'))
    means, covariance = projected(g.get_xyz, g.get_scaling+dscale, g.get_rotation+drot, camera)
    visible = package['visibility_filter']
    assert visible.all()
    assert torch.allclose(means[visible], package['proj_means_2D'][visible], atol=3e-5)
    assert torch.allclose(covariance[visible], symmetric_cov(package['conic_2D_inv'])[visible], atol=5e-5, rtol=2e-5)


def symmetric_cov(v):
    return torch.stack([v[:, 0], v[:, 1], v[:, 1], v[:, 2]], -1).reshape(-1, 2, 2)


@pytest.mark.parametrize('t', [0., .002, .4, .999, 1.])
def test_material_time_stencils(t):
    parameter = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    f = lambda s: parameter*torch.tensor([s**3, s*s, s], dtype=torch.float64)
    v = derivative(f, t, .001)
    a = derivative(f, t, .005, 2)
    assert torch.allclose(v, parameter*torch.tensor([3*t*t, 2*t, 1.], dtype=torch.float64), atol=4e-6)
    assert torch.allclose(a, parameter*torch.tensor([6*t, 2., 0.], dtype=torch.float64), atol=1e-8)
    (v.square().sum()+a.square().sum()).backward()
    assert parameter.grad.isfinite() and parameter.grad.abs()>0


def fixture():
    torch.manual_seed(20)
    x = torch.randn(36, 3, dtype=torch.float64, requires_grad=True)
    anchors = torch.arange(36)
    neighbors = torch.stack([torch.cat([anchors[:i], anchors[i+1:]]) for i in range(36)])
    return x, anchors, neighbors


def test_world_gradient_affine_and_degenerate_diagnostic():
    x, a, n = fixture()
    matrix = torch.tensor([[2., 3., -1.], [-1., .2, 4.]], dtype=x.dtype)
    gradient, valid = fit_gradient(x, x@matrix.T+7, a, n, ridge=0)
    assert valid.all() and torch.allclose(gradient, matrix.expand_as(gradient), atol=1e-12)
    gradient.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    flat = x.detach().clone(); flat[:, 2] = 0
    _, valid = fit_gradient(flat, flat, a, n)
    assert not valid.any()


def test_lagrangian_momentum_manufactured_stress_divergence_and_gradients():
    x, a, n = fixture()
    velocity = x*2
    stress = torch.stack([x[:, 0], 2*x[:, 1], 3*x[:, 2], x[:, 0]*0, x[:, 0]*0, x[:, 0]*0], -1)
    stress.retain_grad()
    acceleration = x*0+torch.tensor([1., 2., 3.], dtype=x.dtype)
    r, valid = residual(x, velocity, stress, acceleration, .2, a, a, a, n, n)
    assert valid.all() and r.abs().max()<1e-4
    (r+1).square().mean().backward()
    assert torch.isfinite(x.grad).all() and stress.grad.abs().sum()>0
    assert torch.allclose(stress_tensor(stress, torch.eye(3, dtype=x.dtype)[None].expand(36, -1, -1)*2, .2),
                          stress_tensor(stress, torch.zeros(36, 3, 3, dtype=x.dtype), .2))


def test_projection_covariance_and_true_pixel_gradient():
    camera = SimpleNamespace(world_view_transform=torch.eye(4, dtype=torch.float64), FoVx=math.pi/2,
        FoVy=math.pi/2, image_width=400, image_height=200)
    x = torch.tensor([[0., 0., 2.]], dtype=torch.float64, requires_grad=True)
    scales = torch.ones(1, 3, dtype=x.dtype)*.1
    means, covariance = projected(x, scales, x.new_tensor([[1., 0, 0, 0]]), camera)
    assert torch.allclose(means, x.new_tensor([[199.5, 99.5]]))
    assert torch.allclose(covariance, x.new_tensor([[[100.3, 0], [0, 25.3]]]))
    means[0, 0].backward()
    assert x.grad[0, 0].item() == pytest.approx(100)  # fx/z, pixels/world unit
    torch.autograd.gradcheck(lambda q: projected(q, scales, q.new_tensor([[1., 0, 0, 0]]), camera)[1], (x.detach().requires_grad_(),))


def test_pixel_flow_identity_translation_scale_and_negative_sentinel():
    dtype = torch.float64
    mean = torch.tensor([[4., 3.], [1000., 1000.]], dtype=dtype, requires_grad=True)
    p = {'gs_per_pixel': torch.tensor([[[0]], [[-1]]]),
         'weight_per_gs_pixel': torch.tensor([[[.8]], [[.9]]], dtype=dtype),
         'conic_2D': torch.tensor([[1., 0, 1.], [1., 0, 1.]], dtype=dtype),
         'proj_means_2D': mean.detach().clone(),
         'x_mu': torch.tensor([[[[2.]], [[1.]]], [[[500.]], [[500.]]]], dtype=dtype)}
    cov = torch.eye(2, dtype=dtype)[None].expand(2, -1, -1)
    identity, valid = pixel_flow(p, mean, cov)
    assert valid.all() and identity.abs().max()<2e-8
    shifted, _ = pixel_flow(p, mean+mean.new_tensor([2., -1.]), cov)
    assert torch.allclose(shifted, mean.new_tensor([[[2., -1.]]]))
    scaled, _ = pixel_flow(p, mean, cov*4)
    assert torch.allclose(scaled, mean.new_tensor([[[2., 1.]]]))
    shifted.sum().backward()
    assert torch.equal(mean.grad, mean.new_tensor([[1., 1.], [0., 0.]]))
    torch.autograd.gradcheck(sqrt_spd, (torch.tensor([[[2., .2], [.2, 1.]]], dtype=dtype, requires_grad=True),))
