import numpy as np
import pytest
import torch
from types import SimpleNamespace
from variants.staged_reconstruction.ordinal_depth.core import pair_bank, ordinal_loss, geometry_only


def test_inverse_depth_order_is_not_silently_reversed():
    pseudo = np.tile(np.linspace(1, 20, 80), (50, 1))
    pairs, signs = pair_bank(pseudo, np.ones_like(pseudo, bool), 9, 100)
    metric = torch.tensor(30-pseudo)
    loss = ordinal_loss(metric, torch.ones_like(metric), torch.tensor(pairs), torch.tensor(signs))
    reversed_loss = ordinal_loss(-metric, torch.ones_like(metric), torch.tensor(pairs), torch.tensor(signs))
    assert loss < .001 and reversed_loss > 1.99


def test_pairs_are_reproducible_and_do_not_consume_global_rng():
    np.random.seed(123)
    before = np.random.get_state()
    p = np.arange(5000, dtype=float).reshape(50, 100)
    a, b = pair_bank(p, np.ones_like(p, bool), 42, 150)
    c, d = pair_bank(p, np.ones_like(p, bool), 42, 150)
    assert np.array_equal(a, c) and np.array_equal(b, d)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_masked_invalid_or_equal_pairs_cannot_enter_supervision():
    p = np.arange(6000, dtype=float).reshape(60, 100)
    mask = np.zeros_like(p, bool); mask[5:40, 20:90] = True
    p[:5] = np.nan
    pairs, signs = pair_bank(p, mask, 18, 200)
    assert mask.ravel()[pairs].all() and np.isfinite(p.ravel()[pairs]).all()
    assert (signs != 0).all()
    with pytest.raises(ValueError):
        pair_bank(np.ones_like(p), mask, 1)


def test_wrong_depth_order_receives_correct_finite_gradient():
    depth = torch.tensor([[1., 1.1]], requires_grad=True)
    loss = ordinal_loss(depth, torch.ones_like(depth), torch.tensor([[0, 1]]), torch.tensor([1.]))
    loss.backward()
    assert depth.grad.isfinite().all()
    assert depth.grad[0, 0] < 0 and depth.grad[0, 1] > 0


def test_geometry_prior_does_not_train_appearance_opacity_scale():
    tensors = [torch.randn(3, 3, requires_grad=True) for _ in range(5)]
    value = SimpleNamespace(get_xyz=tensors[0], get_scaling=tensors[1], get_rotation=tensors[2],
        get_opacity=tensors[3], get_features=tensors[4], max_sh_degree=3, active_sh_degree=3)
    out = geometry_only(value)
    assert out.get_xyz is tensors[0]
    for attr in ['get_scaling', 'get_rotation', 'get_opacity', 'get_features']:
        assert not getattr(out, attr).requires_grad


def test_depth_is_opacity_normalized():
    d = torch.tensor([[1., 2.]])
    a = torch.tensor([[.5, .5]])
    pairs, signs = torch.tensor([[0, 1]]), torch.tensor([-1.])
    assert torch.equal(ordinal_loss(d, a, pairs, signs), ordinal_loss(d*2, a*2, pairs, signs))
