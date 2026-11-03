import pytest
import torch
from variants.static_pool.model import replace_static, dynamic_rows, select_physics_permutation


def fixture_warp(n=6):
    return {k: torch.randn(n, d, dtype=torch.float64, requires_grad=True)
            for k, d in [('d_xyz', 7), ('d_rotation', 4), ('d_scaling', 3)]}


def test_static_is_time_independent_and_dynamic_rows_are_unchanged():
    ids = torch.tensor([1, 4]); a, b = fixture_warp(), fixture_warp()
    reference = {k: torch.randn(2, v.shape[1], dtype=v.dtype) for k, v in a.items()}
    aa, bb = replace_static(a, ids, reference), replace_static(b, ids, reference)
    keep = dynamic_rows(ids, 6)
    for k in a:
        assert torch.equal(aa[k][ids], bb[k][ids])
        assert torch.equal(aa[k][keep], a[k][keep])
        assert torch.equal(bb[k][keep], b[k][keep])
        assert aa[k].data_ptr() != a[k].data_ptr()


def test_static_blocks_deformation_gradients_but_keeps_dynamic_gradients():
    ids = torch.tensor([0, 3]); w = fixture_warp()
    refs = {k: torch.randn(2, v.shape[1], dtype=v.dtype, requires_grad=True) for k, v in w.items()}
    result = replace_static(w, ids, refs)
    sum(v.square().sum() for v in result.values()).backward()
    for k in w:
        assert torch.equal(w[k].grad[ids], torch.zeros_like(w[k].grad[ids]))
        assert w[k].grad[dynamic_rows(ids, 6)].abs().sum() > 0
        assert refs[k].grad is None


def test_static_geometry_remains_learnable_without_time_dependence():
    ids = torch.tensor([1, 4]); w = fixture_warp()
    refs = {k: torch.zeros(2, v.shape[1], dtype=v.dtype) for k, v in w.items()}
    refs['d_xyz'][:, 0] = 1
    out = replace_static(w, ids, refs)
    xyz = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    scale = torch.rand(6, 3, dtype=torch.float64, requires_grad=True)
    # Identity reference rotation: exercise the same canonical/offset dependency.
    center = xyz + out['d_xyz'][:, 4:]
    cov_diag = (scale + out['d_scaling']).square()
    (center[ids].square().sum() + cov_diag[ids].sum()).backward()
    assert xyz.grad[ids].abs().sum() > 0 and scale.grad[ids].abs().sum() > 0


@pytest.mark.parametrize('ids', [torch.tensor([], dtype=torch.long), torch.tensor([-1]),
                               torch.tensor([6]), torch.tensor([1, 1]), torch.tensor([1.])])
def test_invalid_static_rows_are_rejected(ids):
    w = fixture_warp()
    refs = {k: torch.zeros(len(ids), v.shape[1]) for k, v in w.items()}
    with pytest.raises(ValueError): replace_static(w, ids, refs)


def test_invalid_reference_is_rejected():
    w = fixture_warp(); ids = torch.tensor([1])
    refs = {k: torch.zeros(1, v.shape[1]) for k, v in w.items()}
    refs['d_xyz'][0, 0] = float('nan')
    with pytest.raises(ValueError): replace_static(w, ids, refs)


def test_dynamic_cmr_preserves_original_particle_row_mapping():
    ids = torch.tensor([1, 4]); keep = dynamic_rows(ids, 6)
    rows, denominator = select_physics_permutation(keep, 6, 20, 'cpu')
    assert denominator == 4 and sorted(rows.tolist()) == [0, 2, 3, 5]
    particle_embedding_ids = torch.tensor([8, 2, 2, 9, 3, 7])
    assert sorted(particle_embedding_ids[rows].tolist()) == [2, 7, 8, 9]


def test_all_static_has_empty_dynamic_domain():
    keep = dynamic_rows(torch.arange(6), 6)
    rows, denominator = select_physics_permutation(keep, 6, 20, 'cpu')
    assert denominator == 0 and len(rows) == 0


def test_original_sampling_rng_is_unchanged_when_no_pool():
    state = torch.random.get_rng_state()
    a, n = select_physics_permutation(None, 6, 3, 'cpu')
    torch.random.set_rng_state(state)
    b = torch.randperm(6)[:3]
    assert n == 6 and torch.equal(a, b)


def test_disabled_pool_does_not_access_model_or_binding():
    from variants.static_pool.render import pool_for
    assert pool_for(object(), {'enabled': True, 'hard_static_pool': False,
                               'static_pool_binding': 'missing.npz'}) is None
    assert pool_for(object(), {'enabled': False, 'hard_static_pool': True}) is None


def test_runtime_keeps_all_original_objectives():
    from variants.static_pool.runtime import Tuning, Parent
    for name in ['image_loss', 'regularization', 'kinematic_loss', 'physics_time',
                 'time_step', 'after_update', 'evaluate']:
        assert getattr(Tuning, name) is getattr(Parent, name)


def test_disabled_render_returns_identical_objects(monkeypatch):
    from variants.static_pool import render
    monkeypatch.setattr(render, 'pool_for', lambda pc: None)
    a, b, c = object(), object(), object()
    got = render.apply_warp(object(), a, b, c)
    assert all(x is y for x, y in zip(got, (a, b, c)))
