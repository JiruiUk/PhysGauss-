import pytest
import torch
from variants.staged_reconstruction.invertible_motion.core import InvertibleMotion
from variants.staged_reconstruction.covariance_transport.core import motion_with_differential, covariance, transport, pack


def perturbed():
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(728)
        motion = InvertibleMotion(width=8).double()
        with torch.no_grad():
            for layer in motion.layers:
                layer.conditioner[-1].weight.normal_(0, .25)
                layer.conditioner[-1].bias.normal_(0, .2)
    return motion


@pytest.mark.parametrize('time', [0., .43, 1.])
def test_analytic_chain_matches_independent_autograd_and_original_forward(time):
    motion = perturbed()
    x = torch.tensor([[.2, -.3, .7], [.8, .4, -.2]], dtype=torch.double)
    t = x.new_full((2, 1), time)
    moved, j = motion_with_differential(motion, x, t)
    assert torch.equal(moved, motion(x, t))
    for k in range(2):
        expected = torch.autograd.functional.jacobian(lambda value: motion(value[None], t[k:k+1])[0], x[k])
        assert torch.allclose(j[k], expected, atol=1e-12, rtol=1e-12)
    _, logdet = motion(x, t, return_logdet=True)
    assert torch.allclose(torch.linalg.det(j), logdet.exp(), atol=1e-12)


def test_identity_motion_preserves_anisotropic_reference_covariance():
    motion = InvertibleMotion(width=8).double()
    x = torch.tensor([[.2, -.3, .7]], dtype=torch.double)
    moved, j = motion_with_differential(motion, x, x.new_full((1, 1), .43))
    c = covariance(x.new_tensor([[.2, .4, .7]]), x.new_tensor([[1., 0., 0., 0.]]))
    assert torch.equal(x, moved) and torch.equal(transport(j, c), c)


def test_affine_covariance_matches_direct_transformation_including_shear():
    # E[(F z)(F z)^T] == F E[z z^T] F^T, using a deterministic full-rank sample.
    z = torch.tensor([[1., 2., 0.], [-1., -2., 0.], [0., 1., 3.], [0., -1., -3.],
                      [2., 0., 1.], [-2., 0., -1.]], dtype=torch.double)
    f = torch.tensor([[1., .4, .2], [0., .8, .1], [0., 0., 1.2]], dtype=z.dtype)
    canonical = z.T@z/len(z)
    moved = z@f.T
    actual = transport(f[None], canonical[None])[0]
    assert torch.allclose(actual, moved.T@moved/len(z), atol=1e-12)
    assert torch.linalg.eigvalsh(actual).min() > 0


def test_rigid_rotation_transports_axes_and_preserves_covariance_spectrum():
    c = torch.diag(torch.tensor([.04, .16, .49], dtype=torch.double))[None]
    angle = torch.tensor(.7, dtype=torch.double)
    r = torch.tensor([[angle.cos(), -angle.sin(), 0.], [angle.sin(), angle.cos(), 0.], [0., 0., 1.]], dtype=c.dtype)[None]
    actual = transport(r, c)
    assert torch.allclose(torch.linalg.eigvalsh(actual), torch.linalg.eigvalsh(c), atol=1e-12)
    assert torch.allclose(pack(actual), actual[:, [0,0,0,1,1,2], [0,1,2,1,2,2]])


def test_covariance_gradient_includes_coordinate_dependence_of_jacobian():
    motion = perturbed()
    x = torch.tensor([[.2, -.3, .7]], dtype=torch.double, requires_grad=True)
    t = torch.tensor([[.43]], dtype=torch.double, requires_grad=True)
    scale = torch.tensor([[.2, .4, .7]], dtype=torch.double, requires_grad=True)
    q = torch.tensor([[1., .2, -.1, .3]], dtype=torch.double, requires_grad=True)
    def function(a, b, s, rotation):
        _, j = motion_with_differential(motion, a, b)
        return transport(j, covariance(s, torch.nn.functional.normalize(rotation, dim=-1)))
    assert torch.autograd.gradcheck(function, (x, t, scale, q), eps=1e-6, atol=1e-5, rtol=1e-4)
    gradients = torch.autograd.grad(function(x, t, scale, q).square().sum(), [x, t, scale, q, *motion.parameters()])
    assert all(g.isfinite().all() and g.abs().sum() > 0 for g in gradients)


def test_reject_time_shape_that_would_broadcast_across_points():
    with pytest.raises(ValueError):
        motion_with_differential(perturbed(), torch.zeros(2, 3, dtype=torch.double), torch.zeros(1, 1, dtype=torch.double))


def test_disabled_runtime_skips_all_assets_and_keeps_original_methods():
    from types import SimpleNamespace
    from variants.staged_reconstruction.covariance_transport.runtime import install
    original = object()
    scene = SimpleNamespace(dynamic_at=original, render=original, deforms=[original])
    assert install(scene, {}, False) is None
    assert install(scene, {'covariance_transport_enabled':False}, True) is None
    assert scene.dynamic_at is original and scene.render is original and scene.deforms[0] is original


def test_renderer_wrapper_routes_only_marked_banks_and_does_not_mutate_pipe():
    from types import SimpleNamespace
    from variants.staged_reconstruction.covariance_transport.runtime import renderer_for
    calls = []
    def original(cam, bank, pipe, *args, **kwargs):
        calls.append(pipe.compute_cov3D_python)
        return bank
    pipe = SimpleNamespace(compute_cov3D_python=False)
    render = renderer_for(original)
    assert renderer_for(render) is render
    plain = SimpleNamespace()
    marked = SimpleNamespace(transported_covariance=torch.zeros(1, 6))
    assert render(None, plain, pipe) is plain
    assert render(None, marked, pipe) is marked
    assert calls == [False, True] and pipe.compute_cov3D_python is False
