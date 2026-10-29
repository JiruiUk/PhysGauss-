"""Specify one shared effective stiffness before full training integration."""
import pytest
import torch
from pidg_ext.constitutive import NeoHookean
from pidg_ext.deformation_gradient import canonical_neighbors, estimate_F
from pidg_ext.losses import constitutive_consistency
from variants.c1_material.model import SharedYoung, conditional_profile


def test_exactly_one_learnable_scalar_and_fixed_poisson():
    m = SharedYoung(1., .3, learn=True)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 1
    assert m.young.item() == pytest.approx(1.)
    assert m()['mu'].item() == pytest.approx(1 / 2.6)
    assert m()['kappa'].item() == pytest.approx(1 / 1.2)
    assert sum(p.numel() for p in SharedYoung(1., .3).parameters() if p.requires_grad) == 0


@pytest.mark.parametrize('kwargs', [dict(young=0), dict(poisson=.5), dict(poisson=-1),
    dict(young=float('nan')), dict(lower=2.), dict(upper=.1)])
def test_invalid_material_rejected(kwargs):
    with pytest.raises(ValueError): SharedYoung(**kwargs)


def test_positive_bounded_stiffness_without_rng_consumption():
    state = torch.get_rng_state().clone()
    m = SharedYoung(learn=True)
    assert torch.equal(state, torch.get_rng_state())
    for value in (-1000., 1000.):
        with torch.no_grad(): m.raw_E.fill_(value)
        assert .05 <= m.young <= 20


def test_objectivity_and_rigid_rotation_zero_stress():
    m = SharedYoung()
    a = torch.tensor(.7, dtype=torch.float64)
    z = a * 0; o = z + 1
    r = torch.stack([a.cos(),-a.sin(),z,a.sin(),a.cos(),z,z,z,o]).reshape(3,3)
    law = NeoHookean()
    assert law(r[None], m()).abs().max() < 1e-12
    f = torch.diag(torch.tensor([1.2,.9,1.05], dtype=torch.float64))
    assert torch.allclose(law((r@f)[None],m()), r@law(f[None],m())@r.T, atol=1e-12)


def test_affine_F_and_loss_backpropagate_to_trajectory_stress_and_E():
    axis = torch.linspace(-.3,.3,4,dtype=torch.float64)
    x = torch.cartesian_prod(axis,axis,axis)
    a = torch.diag(torch.tensor([1.15,.95,1.03],dtype=torch.float64)).requires_grad_()
    y = x @ a.T
    result = estimate_F(x,y,canonical_neighbors(x,16))
    assert result['valid'].all()
    assert torch.allclose(result['F'],a.expand(len(x),3,3),atol=1e-4)
    material = SharedYoung(learn=True)
    pred = torch.zeros_like(result['F'],requires_grad=True)
    loss,_ = constitutive_consistency(pred,result['F'],material(),result['confidence'],result['valid'])
    loss.backward()
    for grad in (a.grad,pred.grad,material.raw_E.grad):
        assert torch.isfinite(grad).all() and grad.abs().sum()>0


def test_fixed_and_learned_identical_initial_stress():
    f = torch.diag(torch.tensor([1.1,1.,.95],dtype=torch.float64))[None]
    assert torch.equal(NeoHookean()(f,SharedYoung()()),NeoHookean()(f,SharedYoung(learn=True)()))


def test_conditional_profile_never_claims_material_identifiability():
    f = torch.diag(torch.tensor([1.1,1.,1.],dtype=torch.float64))[None]
    m = SharedYoung()
    target = NeoHookean()(f,m())
    profile = conditional_profile(target,f,m,torch.ones(1),torch.ones(1,dtype=torch.bool))
    assert profile['conditional_only'] and profile['data_identifiability']=='not_established'
    assert len(profile['points'])==5
    assert min(profile['points'],key=lambda p:p['loss'])['E']==pytest.approx(1.)
