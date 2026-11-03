"""Sparse shared motion contracts before real GPU training."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from variants.staged_reconstruction.sparse_motion.core import controls_from_canonical, matrix, weights, interpolate, SparseDeform


class Field:
    reg_spatial_able = False
    reg_temporal_able = False

    def __init__(self):
        self.bias = torch.nn.Parameter(torch.tensor(.1))
        self.calls = []

    def step(self, x, t, fixed_attention=False):
        self.calls.append(len(x))
        q = torch.zeros(len(x), 4, dtype=x.dtype); q[:, 0] = 1
        return {'d_xyz': torch.cat([q, torch.sin(x*8+t)*self.bias], -1),
                'd_rotation': torch.zeros_like(q), 'd_scaling': x*0+self.bias,
                'reg_spatial': None, 'reg_temporal': None}


def test_single_rigid_motion_is_reproduced_exactly_including_rotation_of_offsets():
    controls = torch.tensor([[0., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    query = torch.tensor([[.3, .4, .1], [.9, .3, .3]])
    ids, w = weights(query, controls, torch.ones(4))
    q = torch.tensor([np.sqrt(.5), 0, 0, np.sqrt(.5)], dtype=torch.float32).expand(4, 4)
    node = {'d_xyz': torch.cat([q, torch.tensor([.4, .6, .8]).expand(4, 3)], -1),
            'd_rotation': torch.zeros(4, 4), 'd_scaling': torch.ones(4, 3)*.02}
    out = interpolate(query, node, ids, w)
    actual = (matrix(out['d_xyz'][:, :4])@query[..., None]).squeeze(-1)+out['d_xyz'][:, 4:]
    expected = (matrix(q[0])@query.T).T+torch.tensor([.4, .6, .8])
    assert torch.allclose(actual, expected, atol=2e-7)


def test_opposite_quaternion_signs_do_not_cancel_identical_rotations():
    q = torch.tensor([[1., 0, 0, 0], [-1., 0, 0, 0], [1., 0, 0, 0], [-1., 0, 0, 0]])
    query = torch.zeros(1, 3)
    node = {'d_xyz': torch.cat([q, torch.zeros(4, 3)], -1), 'd_rotation': q, 'd_scaling': torch.zeros(4, 3)}
    out = interpolate(query, node, torch.tensor([[0, 1, 2, 3]]), torch.ones(1, 4)/4)
    assert torch.equal(matrix(out['d_xyz'][:, :4]), torch.eye(3)[None])


def test_flags_off_returns_original_values_and_only_original_query():
    field = Field()
    controls = torch.zeros(4, 3)
    adapter = SparseDeform(field, controls, torch.ones(4), mix=0)
    x, t = torch.ones(7, 3), torch.ones(7, 1)*.2
    a, b = adapter.step(x, t), field.step(x, t)
    assert all(torch.equal(a[k], b[k]) for k in ['d_xyz', 'd_rotation', 'd_scaling'])
    assert field.calls == [7, 7]


@pytest.mark.parametrize('mix', [.2, .7, 1.])
def test_chunk_permutation_consistency_and_intended_gradients(mix):
    controls = torch.tensor([[0., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    field = Field()
    adapter = SparseDeform(field, controls, torch.ones(4), mix=mix)
    x = torch.linspace(.1, .9, 21).reshape(7, 3).requires_grad_()
    t = torch.ones(7, 1)*.3
    a = adapter.step(x, t)
    b = torch.cat([adapter.step(v, v.new_full((len(v), 1), .3))['d_xyz'] for v in x.split(3)])
    c = adapter.step(x.flip(0), t)['d_xyz'].flip(0)
    assert torch.allclose(a['d_xyz'], b, atol=1e-6)
    assert torch.allclose(a['d_xyz'], c, atol=1e-6)
    center = (matrix(a['d_xyz'][:, :4])@x[..., None]).squeeze(-1)+a['d_xyz'][:, 4:]
    center.square().mean().backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
    assert field.bias.grad.isfinite() and field.bias.grad.abs() > 0


def test_fps_determinism_fixed_budget_and_no_duplicate_nodes():
    x = np.random.default_rng(12).normal(size=(100, 3)).astype(np.float32)
    a = controls_from_canonical(x, 16)
    b = controls_from_canonical(x, 16)
    assert len(np.unique(a[0])) == 16 and np.array_equal(a[0], b[0])
    assert (a[2] > 0).all() and a[1].shape == (16, 3)


def test_full_sparse_mode_never_queries_dense_point_motion_network():
    field = Field()
    adapter = SparseDeform(field, torch.eye(4, 3), torch.ones(4), mix=1)
    adapter.step(torch.ones(20, 3)*.2, torch.ones(20, 1)*.1)
    assert field.calls == [4]


def test_disabling_regularization_for_physics_delegates_to_same_underlying_field():
    field = Field()
    field.reg_spatial_able = True
    adapter = SparseDeform(field, torch.zeros(4, 3), torch.ones(4))
    adapter.reg_spatial_able = False
    assert not field.reg_spatial_able


def test_gaussian_kernel_weights_sum_to_one_and_stay_finite_far_away():
    query = torch.tensor([[100., 200., 300.]])
    ids, w = weights(query, torch.eye(4, 3), torch.ones(4)*.01)
    assert w.isfinite().all() and torch.allclose(w.sum(-1), torch.ones(1))
