"""Behavioral tests before adding the optional reconstruction regularizer."""
import subprocess
import sys
import numpy as np
import pytest
import torch
from variants.staged_reconstruction.motion_integrity.core import trajectory_graph, strain_loss


def test_curve_graph_rejects_two_surfaces_that_only_touch_temporarily():
    patch = np.array([[0., 0, 0], [.01, 0, 0], [0, .01, 0], [.01, .01, 0]])
    positions = np.stack([np.concatenate([patch, patch+[offset, 0, 0]]) for offset in [.3, .02, -.3]])
    graph = trajectory_graph(positions, k=2, radius=.05)
    assert len(graph['edges']) > 0
    assert np.all((graph['edges'][:, 0] < 4) == (graph['edges'][:, 1] < 4))
    assert len(np.unique(graph['edges'])) == 8


def test_strain_is_zero_for_arbitrary_rigid_motion_and_permutation():
    x = torch.tensor([[0., 0, 0], [1., 0, 0], [0, 1., 0]])
    e = torch.tensor([[0, 1], [0, 2], [1, 2]])
    d = (x[e[:, 1]]-x[e[:, 0]]).norm(dim=-1)
    rotation = torch.tensor([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
    assert strain_loss(x@rotation+5., e, d, torch.ones(3)) == 0
    assert strain_loss(x@rotation+5., e.flip(0), d.flip(0), torch.ones(3)) == 0


def test_stretch_and_collapse_have_finite_gradients_and_descent_reduces_violation():
    for scale in [.2, 2.]:
        x = torch.tensor([[0., 0, 0], [scale, 0, 0]], requires_grad=True)
        e = torch.tensor([[0, 1]])
        d, w = torch.ones(1), torch.ones(1)
        loss = strain_loss(x, e, d, w)
        loss.backward()
        assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
        assert strain_loss(x-.02*x.grad, e, d, w) < loss


def test_moderate_elastic_strain_is_not_forced_to_zero():
    x = torch.tensor([[0., 0, 0], [1.1, 0, 0]], requires_grad=True)
    assert strain_loss(x, torch.tensor([[0, 1]]), torch.ones(1), torch.ones(1)) == 0


def test_loss_rejects_empty_graph_instead_of_claiming_perfect_result():
    with pytest.raises(ValueError, match='empty graph'):
        strain_loss(torch.zeros(2, 3), torch.empty(0, 2, dtype=torch.long), torch.ones(0), torch.ones(0))


def test_graph_determinism_global_rigid_motion_and_fixed_normalizer():
    x = np.random.default_rng(7).normal(size=(12, 3)).astype(np.float32)*.01
    positions = np.stack([x, x+[.3, .1, 0], x+[-.3, 0, .2]])
    a = trajectory_graph(positions)
    b = trajectory_graph(positions+.4)
    assert np.array_equal(a['edges'], b['edges'])
    assert np.allclose(a['rest'], b['rest'], atol=1e-7)
    p = torch.from_numpy(x*2)
    args = [torch.from_numpy(a[k]) for k in ['edges', 'rest', 'weight']]
    loss = strain_loss(p, *args)
    assert torch.allclose(loss, strain_loss(p, args[0], args[1], args[2]*.2))


def test_disabled_entry_does_not_initialize_cuda_or_read_checkpoint():
    result = subprocess.run([sys.executable, '-m', 'variants.staged_reconstruction.motion_integrity.prepare'], capture_output=True, text=True)
    assert result.returncode != 0 and 'opt-in' in result.stderr
    assert 'ninja' not in result.stdout
