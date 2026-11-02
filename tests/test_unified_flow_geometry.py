import math
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from variants.unified_flow.camera_contract import loader_fovs
from variants.unified_flow.flow import pixel_flow, contribution_weights


def package():
    return {'gs_per_pixel': torch.tensor([[[0]], [[1]], [[-1]]]),
            'conic_2D': torch.tensor([[1., 0, 1.], [1., 0, 1.]], dtype=torch.float64),
            'proj_means_2D': torch.tensor([[4., 3.], [4., 3.]], dtype=torch.float64),
            'x_mu': torch.tensor([[[[2.]], [[1.]]], [[[2.]], [[1.]]], [[[0.]], [[0.]]]], dtype=torch.float64)}


def test_expanding_ellipse_moves_pixel_away_from_center():
    p = package()
    # Pixel=(2,2), mean=(4,3), scale2 -> new pixel=(0,1), flow=(-2,-1).
    mean = p['proj_means_2D'].clone().requires_grad_()
    covariance = 4*torch.eye(2, dtype=mean.dtype)[None].expand(2, -1, -1)
    flow, seen = pixel_flow(p, mean, covariance, mean.new_tensor([.8, .4]))
    assert seen.all() and torch.allclose(flow, mean.new_tensor([[[-2., -1.]]]), atol=1e-7)
    flow.sum().backward()
    assert mean.grad.isfinite().all() and torch.allclose(mean.grad.sum(0), mean.new_ones(2))


def test_weights_use_pre_update_transmittance_and_ignore_sentinel():
    p = package(); p['x_mu'].zero_()
    _, weights = contribution_weights(p, torch.tensor([.8, .4], dtype=torch.float64))
    assert torch.allclose(weights[:, 0, 0], weights.new_tensor([.8, .08, 0.]))
    # CUDA's recorded weights would instead be [.16, .048], biasing mixtures.


def test_camera_fovs_preserve_real_loader_assignment():
    original = .9
    x, y = loader_fovs(original)
    assert x == 2*math.atan(math.tan(original/2)*900/1600) and y == original
    root = Path(__file__).resolve().parents[1]
    loader = (root/'variants/flow_comparison/code/scene_PIDG/dataset_readers.py').read_text(encoding='utf-8')
    assert 'FovY, FovX = fovx, fovy' in loader


def test_actual_loader_camera_info_matches_auxiliary_contract():
    root = Path(__file__).resolve().parents[1]
    source = root/'data/derived/kinematic_resplit/training'
    metadata = json.loads((source/'transforms_test.json').read_text())
    path = root/'variants/flow_comparison/code/scene_PIDG/dataset_readers.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'readCamerasFromTransforms')
    build = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == 'build_cam_info')
    namespace = dict(path=str(source), extension='.png', white_background=False, Image=Image, np=np,
        os=os, Path=Path, CameraInfo=SimpleNamespace,
        fov2focal=lambda a, w: w/(2*math.tan(a/2)), focal2fov=lambda f, h: 2*math.atan(h/(2*f)))
    exec(compile(ast.Module(body=[build], type_ignores=[]), str(path), 'exec'), namespace)
    info = namespace['build_cam_info'](metadata['frames'][0], metadata['camera_angle_x'], False)
    assert (info.FovX, info.FovY) == loader_fovs(metadata['camera_angle_x'])
    # Source RGBA is 2560x1440; the training loader downsamples to 1600x900.
    assert info.width/info.height == 1600/900


@pytest.mark.skipif(os.environ.get('PIDG_UNIFIED_GPU_TEST') != '1', reason='explicit CUDA metadata audit')
def test_real_cuda_metadata_offset_and_compositing_weights(monkeypatch):
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'variants/flow_comparison/code'))
    monkeypatch.delenv('PIDG_METRIC_CONFIG', raising=False)
    monkeypatch.setenv('PIDG_HASH_SOURCE_DIR', str(root/'third_party/PIDG/hashencoder'))
    monkeypatch.setenv('PIDG_HASH_BUILD_DIR', str(root/'third_party/PIDG/tmp_build'))
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.cameras import Camera
    from gaussian_renderer import render
    g = GaussianModel(3)
    g._xyz = torch.nn.Parameter(torch.zeros(1, 3, device='cuda'))
    g._scaling = torch.nn.Parameter(torch.full((1, 3), -2., device='cuda'))
    g._rotation = torch.nn.Parameter(torch.tensor([[1., 0, 0, 0]], device='cuda'))
    g._opacity = torch.nn.Parameter(torch.logit(torch.tensor([[.8]], device='cuda')))
    camera = Camera(0, np.eye(3), np.array([0., 0., 3.]), .8, .8,
        torch.zeros(3, 16, 16), None, 'audit', 0, fid=.5)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    with torch.no_grad():
        p = render(camera, g, pipe, torch.zeros(3, device='cuda'), 0., 0., 0.,
            override_color=torch.ones(1, 3, device='cuda'))
        _, weights = contribution_weights(p, g.get_opacity)
        assert torch.allclose(weights.sum(0), p['render'][0], atol=2e-6)
        alpha = p['render'][0]
        assert torch.allclose(p['weight_per_gs_pixel'].sum(0), alpha*(1-alpha), atol=2e-6)
        y, x = torch.where(p['gs_per_pixel'][0] == 0)
        offsets = p['x_mu'][0, :, y, x].T
        expected = p['proj_means_2D'][0]-torch.stack([x, y], -1)
        assert torch.allclose(offsets, expected)
