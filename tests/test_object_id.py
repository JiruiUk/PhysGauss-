import io
import torch
import pytest
from variants.object_id.model import ObjectMotion,rotation


def scene(learn=True):
    torch.manual_seed(1)
    x=torch.randn(80,3);model=ObjectMotion(4,learn);model.initialize(x)
    warp=torch.zeros(80,7);warp[:,0]=1
    return model,x,torch.full((80,1),.4),warp


def test_blend_zero_returns_original_without_identity_query():
    model=ObjectMotion(4);warp=torch.randn(20,7)
    actual,state=model(torch.zeros(20,3),torch.zeros(20,1),warp,0)
    assert actual is warp and state is None


def test_zero_initial_motion_is_identity_and_soft_assignments_partition():
    model,x,t,warp=scene();p=model.assignments(x)
    assert torch.allclose(p.sum(-1),torch.ones(len(x)))
    assert torch.allclose(model.grouped(x,t,p),x,atol=1e-6)


def test_projection_reconstruction_reaches_identity_and_shared_motion():
    model,x,t,warp=scene()
    with torch.no_grad():model.motion[-1].bias.copy_(torch.linspace(-.2,.3,28))
    output,state=model(x,t,warp,.25)
    positions=torch.einsum('nij,nj->ni',rotation(output[:,:4]),x)+output[:,4:]
    positions[:,:2].square().mean().backward()
    assert model.identity[-1].weight.grad.norm()>0
    assert model.motion[-1].weight.grad.norm()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_frozen_identity_receives_no_gradient_but_motion_does():
    model,x,t,warp=scene(False)
    result,state=model(x,t,warp,.25)
    loss,stats=model.regularizer(state)
    (result[:,4:].square().mean()+loss).backward()
    assert all(p.grad is None for p in model.identity.parameters())
    assert model.motion[-1].weight.grad is not None


def test_checkpoint_reconstructs_assignments_and_warp_exactly():
    model,x,t,warp=scene();buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    other=ObjectMotion(4);other.load_state_dict(torch.load(buffer,weights_only=True))
    assert torch.equal(model(x,t,warp,.3)[0],other(x,t,warp,.3)[0])


def test_children_inherit_field_identity_and_pruning_does_not_require_point_ids():
    model,x,t,warp=scene()
    assert torch.equal(model.assignments(x[:2]),model.assignments(x[[0,1,0]])[:2])
    mass=model.prune_unused(x,min_mass=.3)
    assert 1<=int(model.active.sum())<=model.capacity
    assert torch.allclose(model.assignments(x[::2]).sum(-1),torch.ones(40))
