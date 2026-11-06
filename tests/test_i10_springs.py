import math
import pytest
import torch
from variants.staged_reconstruction.i10_springs.core import median_rest, spring_terms, combine
from variants.staged_reconstruction.i10_revocable_surface.core import transition


def fixture():
    x = torch.tensor([[0.,0,0],[.02,0,0],[0,.03,0],[.02,.03,.004]], dtype=torch.float64)
    edges = [(0,1),(0,2),(1,3),(2,3),(0,3),(1,2)]
    rest = [float((x[j]-x[i]).norm()) for i,j in edges]
    return x,edges,rest


def test_default_off_returns_original_object_without_evaluating_terms():
    base=torch.tensor(2., requires_grad=True); result,aux=combine(base)
    assert result is base and aux==0


def test_reference_is_fixed_fit_median_not_time_specific():
    assert median_rest([[1,2],[2,4],[3,6],[4,8],[5,10],[6,12]]) == [3.5,7.]


@pytest.mark.parametrize('value', [0., float('nan'), -1.])
def test_bad_rest_is_unknown_not_silent_clamp(value):
    with pytest.raises(ValueError): median_rest([[value]])


def test_true_rest_energy_is_zero():
    x,e,l=fixture(); assert spring_terms(x,e,l)['energy'].abs()<1e-14


def test_rigid_transform_preserves_energy():
    x,e,l=fixture();x=x*1.1
    r=x.new_tensor([[0,-1,0],[1,0,0],[0,0,1]])
    assert torch.allclose(spring_terms(x,e,l)['energy'],spring_terms(x@r.T+3,e,l)['energy'],atol=1e-12)


def test_stretch_and_compression_have_symmetric_normalized_penalty():
    x,e,l=fixture();a=spring_terms(x*1.2,e,l);b=spring_terms(x*.8,e,l)
    assert a['energy']>0 and torch.allclose(a['energy'],b['energy'],atol=1e-12)


def test_collapse_is_finite_but_penalized():
    x,e,l=fixture();a=spring_terms(x*0,e,l);assert torch.isfinite(a['energy']) and a['energy']>0


def test_energy_has_intended_finite_position_gradients():
    x,e,l=fixture();x=(x*1.1).requires_grad_();spring_terms(x,e,l)['energy'].backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum()>0
    assert torch.allclose(x.grad.sum(0),torch.zeros(3,dtype=x.dtype),atol=1e-12)


def test_shared_stiffness_is_fixed_not_a_parameter_bank():
    x,e,l=fixture()
    with pytest.raises(ValueError): spring_terms(x,e,l,torch.ones(len(e)))
    assert torch.allclose(spring_terms(x*1.1,e,l,2.)['energy'],2*spring_terms(x*1.1,e,l,1.)['energy'])


def test_unknown_empty_edges_produce_no_spring():
    x,_,_=fixture();assert not spring_terms(x,[],[])['valid'] and spring_terms(x,[],[])['energy']==0


def test_confidence_receives_no_gradient():
    x,e,l=fixture();x=(x*1.1).requires_grad_();conf=torch.tensor(.5,requires_grad=True)
    loss,_=combine(x.sum()*0,spring_terms(x,e,l),confidence=conf,enabled=True);loss.backward()
    assert conf.grad is None and x.grad.abs().sum()>0


def test_gradcheck():
    x,e,l=fixture();x=(x*1.1).requires_grad_()
    assert torch.autograd.gradcheck(lambda y:spring_terms(y,e,l)['energy'],(x,))
