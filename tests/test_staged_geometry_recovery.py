import math
import torch
from variants.staged_reconstruction.geometry_recovery.model import bounded_scale


def test_disabled_scale_policy_exactly_preserves_additive_PIDG_formula():
    base=torch.tensor([[-3.,-4.,-5.]]);delta=torch.tensor([[.1,-.2,.3]])
    assert torch.equal(bounded_scale(base,delta,.01),base.exp()+delta)


def test_identity_scale_initialization_is_preserved():
    spacing=.013
    base=torch.full((3,3),math.log(spacing*.65));delta=torch.zeros_like(base)
    assert torch.allclose(bounded_scale(base,delta,spacing,True),torch.full_like(base,spacing*.65),atol=1e-8)


def test_negative_additive_head_cannot_kill_whole_group_and_large_head_cannot_inflate_it():
    base=torch.full((3,3),-4.,requires_grad=True);delta=torch.tensor([[-.1,0.,.1]]*3,requires_grad=True)
    scale=bounded_scale(base,delta,.013,True);scale.sum().backward()
    assert (scale>=.013*.15).all() and (scale<=.013*2.5).all()
    assert (delta.grad>0).all() and (base.grad>0).all()
    for d in [-100.,100.]:
        extreme=bounded_scale(base.detach(),torch.full_like(base,d),.013,True)
        assert extreme.isfinite().all() and (extreme>=.013*.15).all() and (extreme<=.013*2.5+1e-8).all()
