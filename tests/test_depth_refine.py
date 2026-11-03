import torch
from variants.staged_reconstruction.depth_refine.core import quadratic_offset,accept_update,refine


def test_known_quadratic_vertex():
    grid = torch.tensor([-1.,0.,1.])[:,None]
    target = torch.tensor([-.35,.2,0.])
    offset,valid = quadratic_offset((grid-target)**2+2)
    assert valid.all() and torch.allclose(offset,target,atol=1e-6)


def test_half_cell_bound():
    grid = torch.tensor([-1.,0.,1.])[:,None]
    offset,valid = quadratic_offset((grid-torch.tensor([-2.,2.]))**2)
    assert valid.all() and torch.equal(offset,torch.tensor([-.5,.5]))


def test_degenerate_and_nonfinite_preserve():
    cost = torch.tensor([[1.,1.,float('nan'),1.],[1.,2.,1.,1.],[1.,1.,1.,float('inf')]])
    offset,valid = quadratic_offset(cost)
    assert not valid.any() and torch.equal(offset,torch.zeros(4))


def test_flags_off_exact_object():
    old = torch.tensor([4.,5.])
    actual,mask = refine(None,old,None,None,None,None,None,None,None,None)
    assert actual is old and not mask.any()


def test_any_held_source_worse_refuses_even_when_other_better():
    old,new = torch.tensor([4.,5.]),torch.tensor([4.1,5.1])
    actual,accepted = accept_update(old,new,torch.ones(2),torch.zeros(2),
        torch.tensor([[.3,.3],[.2,.2]]),torch.tensor([[.1,.1],[.21,.19]]),torch.ones(2,dtype=torch.bool),True)
    assert torch.equal(accepted,torch.tensor([False,True]))
    assert actual[0] == old[0] and actual[1] == new[1]


def test_fit_must_improve_and_nonfinite_cannot_pass():
    old,new = torch.tensor([4.,5.,6.]),torch.tensor([4.1,float('nan'),6.1])
    actual,accepted = accept_update(old,new,torch.ones(3),torch.tensor([1.01,0.,0.]),
        torch.ones(2,3),torch.zeros(2,3),torch.tensor([True,True,False]),True)
    assert not accepted.any() and torch.equal(actual,old)


def test_acceptance_flags_off_exact():
    old = torch.tensor([4.])
    actual,accepted = accept_update(old,old+1,old,old,old[None],old[None],torch.ones(1,dtype=torch.bool))
    assert actual is old and not accepted.any()


def test_zero_step_is_not_counted_as_update():
    old = torch.tensor([4.])
    actual,accepted = accept_update(old,old,old,old,old[None],old[None],torch.ones(1,dtype=torch.bool),True)
    assert torch.equal(actual,old) and not accepted.any()
