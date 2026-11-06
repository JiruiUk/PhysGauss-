import torch
from physgauss.config import TrainConfig
from physgauss.models import PhysGaussScene
from physgauss.models.gaussians import GaussianState
from physgauss.rendering import Camera, ReferenceRenderer
from physgauss.losses import visual_losses


def test_render_gradients_reach_gaussian_and_deformation_parameters(dataset):
    scene = PhysGaussScene(dataset.xyz, dataset.colors, dataset.scales, TrainConfig())
    state, variance = scene(0.4)
    result = ReferenceRenderer().render(state, dataset.camera(0), variance)
    objective = result.rgb.square().mean()+result.depth.mean()+result.variance.mean()
    objective.backward()
    for parameter in scene.canonical.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert scene.deformation.field.net[-1].weight.grad.abs().sum() > 0
    assert scene.uncertainty.field.net[-1].weight.grad.abs().sum() > 0


def test_canonical_time_is_exact_identity(dataset):
    scene = PhysGaussScene(dataset.xyz, dataset.colors, dataset.scales, TrainConfig(canonical_time=0.3))
    initial = scene.canonical()
    dynamic, _ = scene(0.3)
    torch.testing.assert_close(dynamic.means, initial.means)
    torch.testing.assert_close(dynamic.scales, initial.scales)
    torch.testing.assert_close(dynamic.quaternions, initial.quaternions)


def test_occlusion_depth_and_camera_motion_projection():
    K = torch.tensor([[10., 0, 4], [0, 10., 4], [0, 0, 1]])
    camera = Camera(K, torch.eye(4), 9, 9)
    next_transform = torch.eye(4)
    next_transform[0, 3] = 0.2
    next_camera = Camera(K, next_transform, 9, 9)
    state = GaussianState(torch.tensor([[0., 0., 2.], [0., 0., 4.]]), torch.full((2, 3), 0.2),
                          torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                          torch.tensor([[1., 0., 0.], [0., 0., 1.]]), torch.full((2, 1), 0.9))
    result = ReferenceRenderer().render(state, camera, torch.full((2, 1), 0.01), state, next_camera)
    torch.testing.assert_close(result.rgb[4, 4], torch.tensor([0.9, 0., 0.09]))
    torch.testing.assert_close(result.depth[4, 4], torch.tensor((0.9*2+0.09*4)/0.99))
    torch.testing.assert_close(result.flow[4, 4], torch.tensor([(0.9+0.09*0.5)/0.99, 0.]))


def test_zero_flow_error_has_finite_gradient(dataset):
    frame = dataset.frame(0)
    scene = PhysGaussScene(dataset.xyz, dataset.colors, dataset.scales, TrainConfig())
    state, variance = scene(0.1)
    result = ReferenceRenderer().render(state, frame.camera, variance, state, frame.camera)
    frame.flow = torch.zeros_like(result.flow)
    losses = visual_losses(result, frame)
    losses["flow"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in scene.parameters())
