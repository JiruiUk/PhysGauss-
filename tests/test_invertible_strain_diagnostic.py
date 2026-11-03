"""Mathematical checks for a read-only map diagnostic, not a material model."""
import torch

from scripts.audit_invertible_strain import pointwise_jacobian, relative_gradient
from variants.staged_reconstruction.invertible_motion.core import InvertibleMotion


def test_batch_jacobians_match_full_autograd_for_nonlinear_independent_points():
    x = torch.tensor([[.2, -.3, .7], [.8, .4, -.2]], dtype=torch.double)
    def mapping(value):
        a, b, c = value.unbind(-1)
        return torch.stack([a*b, b.sin()+c*c, c.exp()+a], -1)
    actual = pointwise_jacobian(mapping, x)
    full = torch.autograd.functional.jacobian(mapping, x)
    for i in range(len(x)):
        assert torch.allclose(actual[i], full[i, :, i, :], atol=1e-12)
        assert torch.count_nonzero(full[i, :, 1-i, :]) == 0
    assert not actual.requires_grad


def test_rigid_rotation_has_unit_principal_stretches():
    angle = torch.tensor(.7, dtype=torch.double)
    r = torch.tensor([[angle.cos(), -angle.sin(), 0.],
                      [angle.sin(), angle.cos(), 0.], [0., 0., 1.]], dtype=torch.double)
    x = torch.tensor([[.2, -.3, .7], [.8, .4, -.2]], dtype=torch.double)
    j = pointwise_jacobian(lambda value: value@r.T+.4, x)
    assert torch.allclose(j, r.expand(2, 3, 3), atol=1e-12)
    assert torch.allclose(torch.linalg.svdvals(j), torch.ones(2, 3, dtype=x.dtype), atol=1e-12)


def test_positive_unit_determinant_does_not_bound_shear():
    shear = torch.tensor([[1., 12., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.double)
    j = pointwise_jacobian(lambda value: value@shear.T, torch.ones(1, 3, dtype=torch.double))
    s = torch.linalg.svdvals(j)
    assert torch.allclose(torch.linalg.det(j), torch.ones(1, dtype=j.dtype))
    assert s[0, -1] < .1 and s[0, 0] > 10


def test_relative_gradient_uses_current_times_inverse_reference_in_correct_order():
    a = torch.tensor([[[2., 1., 0.], [0., 1., .2], [0., 0., .7]]], dtype=torch.double)
    b = torch.tensor([[[1., .1, 0.], [0., 2., .3], [.2, 0., 1.]]], dtype=torch.double)
    f = relative_gradient(a, b)
    assert torch.allclose(f@b, a, atol=1e-12)
    assert not torch.allclose(f, torch.linalg.solve(b, a))
    assert torch.allclose(relative_gradient(b, b), torch.eye(3, dtype=b.dtype)[None], atol=1e-12)


def test_coupling_jacobian_matches_finite_difference_and_analytic_logdet():
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(918)
        motion = InvertibleMotion(width=12).double()
        with torch.no_grad():
            for layer in motion.layers:
                layer.conditioner[-1].weight.normal_(0, .3)
                layer.conditioner[-1].bias.normal_(0, .1)
    x = torch.tensor([[.2, -.3, .7], [.8, .4, -.2]], dtype=torch.double)
    t = x.new_full((2, 1), .43)
    j = pointwise_jacobian(lambda value: motion(value, t), x)
    eps = 1e-5
    finite = torch.stack([(motion(x+torch.eye(3, dtype=x.dtype)[k]*eps, t)-
                           motion(x-torch.eye(3, dtype=x.dtype)[k]*eps, t))/(2*eps)
                          for k in range(3)], -1)
    assert torch.allclose(j, finite, atol=2e-8, rtol=1e-7)
    _, logdet = motion(x, t, return_logdet=True)
    assert torch.allclose(torch.linalg.det(j), logdet.exp(), atol=1e-12)
