import math
import pytest
import torch
from variants.staged_reconstruction.i10_surface_support.core import (
    covariance, support_spectrum, geometry_terms, footprint, coverage_term, combine)

C = dict(lower_ratio=.8, upper_ratio=1.25, edge_lower_ratio=.5, sigma_support=2.)


def fixture():
    xyz = torch.tensor([[0., 0, 0], [.02, 0, .001], [0, .025, -.001], [.02, .025, 0]], dtype=torch.float64)
    scales = torch.tensor([[.003, .004, .001]]*4, dtype=torch.float64)
    q = torch.tensor([[1., 0, 0, 0]]*4, dtype=torch.float64)
    cov = covariance(scales, q)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2)]
    ref = dict(spectrum=support_spectrum(xyz, cov).tolist(), max_std=scales.max(-1).values.tolist(),
               edge_lengths=[float(torch.linalg.vector_norm(xyz[j]-xyz[i])) for i, j in edges])
    return xyz, cov, edges, ref


def test_covariance_rotation_normalizes_quaternion():
    s = torch.tensor([[1., 2, 3]])
    q = torch.tensor([[math.sqrt(.5), 0, 0, math.sqrt(.5)]])*7
    assert torch.allclose(covariance(s, q)[0], torch.diag(torch.tensor([4., 1, 9])), atol=1e-5)


def test_rigid_motion_preserves_connection_thickness_and_area():
    x, cov, edges, ref = fixture()
    r = torch.tensor([[0., -1, 0], [1., 0, 0], [0., 0, 1]], dtype=x.dtype)
    a = geometry_terms(x, cov, edges, ref, C)
    b = geometry_terms(x@r.T+3, r@cov@r.T, edges, ref, C)
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-10)


def test_reference_has_zero_thickness_area_penalty():
    x, cov, edges, ref = fixture()
    result = geometry_terms(x, cov, edges, ref, C)
    assert result['thickness']==0 and result['area_scale']==0
    assert result['connection']>0


def test_thickness_inflation_and_collapse_both_penalized():
    x, cov, edges, ref = fixture()
    for factor in [.01, 10.]:
        transform = torch.diag(x.new_tensor([1., 1, factor]))
        result = geometry_terms(x@transform, transform@cov@transform, edges, ref, C)
        assert result['thickness']>0


def test_gap_penalty_backpropagates_to_positions_and_scales():
    x, cov, edges, ref = fixture()
    x.requires_grad_(); cov.requires_grad_()
    result = geometry_terms(x, cov, edges, ref, C)
    result['connection'].backward()
    for value in [x, cov]:
        assert value.grad.isfinite().all() and value.grad.abs().sum()>0


def test_area_collapse_and_unbounded_gaussian_growth_penalized():
    x, cov, edges, ref = fixture()
    assert geometry_terms(x*.1, cov*.01, edges, ref, C)['area_scale']>0
    assert geometry_terms(x, cov*100, edges, ref, C)['area_scale']>0


def test_duplicate_centers_have_finite_loss_and_gradient():
    x, cov, edges, ref = fixture();x = torch.zeros_like(x, requires_grad=True)
    loss = geometry_terms(x, cov, edges, ref, C)
    sum(loss[k] for k in ['connection', 'thickness', 'area_scale']).backward()
    assert x.grad.isfinite().all()


def test_native_footprint_bilinear_fractional_coordinates():
    field = torch.arange(100, dtype=torch.float64).reshape(10, 10)
    assert torch.allclose(footprint(field, [4.5, 4.5]), field.new_tensor(49.5))


def test_missing_footprint_is_unknown_not_coverage_target():
    field = torch.zeros(10, 10, requires_grad=True)
    value, known = coverage_term(field, [0, 0], .5)
    assert not known and value==0


def test_coverage_does_not_require_full_image_alpha_one():
    field = torch.full((10, 10), .1, requires_grad=True)
    value, known = coverage_term(field, [5, 5], .1)
    assert known and value==0
    value, known = coverage_term(field, [5, 5], .5)
    assert known and value>0
    value.backward();assert field.grad.isfinite().all() and field.grad.abs().sum()>0


def test_disabled_path_exact_object_identity_and_no_inputs():
    base = torch.tensor(1., requires_grad=True)
    value, auxiliary = combine(base, None)
    assert value is base and auxiliary==0


def test_unknown_zero_confidence_suspends_all_structural_terms():
    base = torch.tensor(1., requires_grad=True)
    value, auxiliary = combine(base, dict(connection=2., thickness=3., area_scale=4., coverage=5.), confidence=0., enabled=True)
    assert value==base and auxiliary==0


def test_unknown_parent_coverage_reference_is_not_zero_error_target():
    field = torch.zeros(10, 10, requires_grad=True)
    value, known = coverage_term(field, [5, 5], None)
    assert not known and value==0
