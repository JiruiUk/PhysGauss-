import torch

from scripts.audit_render_covariance import raster_covariance, covariance_axis_ratio


def test_unit_quaternion_rotates_covariance_without_changing_principal_axes():
    s = torch.tensor([[.2, .5, .8]], dtype=torch.double)
    angle = torch.tensor(.7, dtype=torch.double)
    q = torch.tensor([[angle.cos(), 0., 0., angle.sin()]], dtype=torch.double)
    c = raster_covariance(s, q)
    assert torch.allclose(torch.linalg.eigvalsh(c).sqrt(), s, atol=1e-12)
    assert torch.allclose(c, c.transpose(-1, -2), atol=1e-12)


def test_raw_nonunit_quaternion_is_not_silently_normalized():
    s = torch.ones(1, 3, dtype=torch.double)
    q = torch.tensor([[0., 2., 0., 0.]], dtype=torch.double)
    assert torch.equal(raster_covariance(s, q), torch.diag(torch.tensor([1., 49., 49.], dtype=q.dtype))[None])
    assert torch.equal(raster_covariance(s, q/q.norm(dim=-1, keepdim=True)), torch.eye(3, dtype=q.dtype)[None])


def test_covariance_ratio_detects_axis_shrink_and_expansion_and_is_basis_invariant():
    ref = torch.diag(torch.tensor([.2, .5, .8], dtype=torch.double))[None]
    predicted = ref*4
    assert torch.allclose(covariance_axis_ratio(predicted, ref), torch.full((1, 3), 2., dtype=ref.dtype), atol=1e-12)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=ref.dtype)
    a = torch.diag(torch.tensor([.05, .5, 3.2], dtype=ref.dtype))[None]
    expected = torch.tensor([[.5, 1., 2.]], dtype=ref.dtype)
    assert torch.allclose(covariance_axis_ratio(a, ref), expected, atol=1e-12)
    assert torch.allclose(covariance_axis_ratio(rotation@a@rotation.T, rotation@ref@rotation.T), expected, atol=1e-12)
