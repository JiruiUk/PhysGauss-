"""Tests precede native training integration of the optional I9 prior."""
import math
import pytest
import torch
from variants.staged_reconstruction.relative_strain.core import (
    relative_differential, log_stretch_energy, sample_indices, strain_prior)
from variants.staged_reconstruction.invertible_motion.core import InvertibleMotion


def rotation():
    angle = .73
    return torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                         [math.sin(angle), math.cos(angle), 0], [0, 0, 1]], dtype=torch.float64)


def test_affine_relative_map_and_common_canonical_gauge():
    ref = torch.tensor([[[1.2, .2, -.1], [0., .8, .1], [.1, 0., 1.1]]], dtype=torch.float64)
    target = rotation()[None] @ torch.diag(torch.tensor([1.5, .6, 1.2], dtype=torch.float64))[None]
    current = target @ ref
    torch.testing.assert_close(relative_differential(current, ref), target)
    gauge = torch.tensor([[[.7, .3, 0.], [0., 1.4, .1], [.1, 0., .9]]], dtype=torch.float64)
    torch.testing.assert_close(relative_differential(current@gauge, ref@gauge), target)


def test_rigid_relative_rotation_has_zero_energy():
    robust, energy, _ = log_stretch_energy(rotation()[None])
    assert energy.max() < 1e-28 and robust.max() < 1e-26


def test_rotation_and_contraction_expansion_symmetry():
    stretch = torch.diag(torch.tensor([2., .8, 1.3], dtype=torch.float64))[None]
    robust, energy, _ = log_stretch_energy(stretch)
    inverse = log_stretch_energy(torch.linalg.inv(stretch))
    transformed = log_stretch_energy(rotation()[None] @ stretch @ rotation().T[None])
    torch.testing.assert_close(robust, inverse[0]); torch.testing.assert_close(energy, inverse[1])
    torch.testing.assert_close(robust, transformed[0])
    expected = torch.tensor([2., .8, 1.3], dtype=torch.float64).log().square().sum()
    torch.testing.assert_close(energy[0], expected)


def test_finite_gradients_to_both_configurations_and_gradcheck():
    current = torch.tensor([[[1.6, .1, 0.], [.2, .8, .1], [0., .1, 1.1]]], dtype=torch.float64, requires_grad=True)
    ref = torch.tensor([[[1.1, .1, 0.], [0., .9, .1], [.1, 0., 1.2]]], dtype=torch.float64, requires_grad=True)
    def loss(a, b):
        return log_stretch_energy(relative_differential(a, b))[0].mean()
    assert torch.autograd.gradcheck(loss, (current, ref), eps=1e-6, atol=2e-5)
    for value in torch.autograd.grad(loss(current, ref), (current, ref)):
        assert value.isfinite().all() and value.abs().sum() > 0


def test_identity_backward_is_finite():
    identity = torch.eye(3, dtype=torch.float64)[None].requires_grad_()
    loss = log_stretch_energy(identity)[0].mean()
    gradient, = torch.autograd.grad(loss, identity)
    assert gradient.isfinite().all() and torch.count_nonzero(gradient) == 0


def test_deterministic_samples_cover_all_rows_without_rng():
    before = torch.get_rng_state().clone()
    seen = torch.cat([sample_indices(31, 8, step, 'cpu') for step in range(1, 5)])
    assert set(seen.tolist()) == set(range(31))
    assert torch.equal(before, torch.get_rng_state())
    assert len(sample_indices(3, 512, 1, 'cpu')) == 3


def test_actual_shared_map_prior_backpropagates_and_preserves_rng():
    torch.manual_seed(19)
    motion = InvertibleMotion(12, .2, .2).double()
    with torch.no_grad():
        for layer in motion.layers:
            layer.conditioner[-1].weight.normal_(0., .1)
    points = torch.randn(17, 3, dtype=torch.float64, requires_grad=True)
    before = torch.get_rng_state().clone()
    loss, row = strain_prior(motion, points, .43, 7,
                            {'strain_samples': 9, 'strain_reference_time': 1/199, 'strain_robust_scale': .3})
    gradients = torch.autograd.grad(loss, [points, *motion.parameters()], allow_unused=True)
    assert torch.equal(before, torch.get_rng_state()) and row['sample_count'] == 9
    assert gradients[0].isfinite().all() and gradients[0].abs().sum() > 0
    assert all(g is None or g.isfinite().all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients[1:] if g is not None) > 0


def test_invalid_arguments_and_singular_map_are_rejected():
    with pytest.raises(ValueError): sample_indices(0, 8, 1, 'cpu')
    with pytest.raises(ValueError): log_stretch_energy(torch.eye(3)[None], 0.)
    with pytest.raises(ValueError): relative_differential(torch.eye(3), torch.eye(3))
    with pytest.raises(FloatingPointError): log_stretch_energy(torch.zeros(1, 3, 3))
