import pytest
import torch
from physics.kinematics import compute_velocity_acceleration
from physics.hash_autograd import interpolate_hash


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_quadratic_harmonic_and_linear(dtype):
    X=torch.randn(4,3,dtype=dtype)
    coefficient=torch.tensor([.2,.4,-.3],dtype=dtype,requires_grad=True)
    x,v,a=compute_velocity_acceleration(lambda X,t:X+coefficient*t*t,X,.2357)
    torch.testing.assert_close(v, (2*coefficient*.2357).expand_as(X))
    torch.testing.assert_close(a,(2*coefficient).expand_as(X))
    a.square().sum().backward()
    assert coefficient.grad is not None and coefficient.grad.norm()>0
    _,_,a=compute_velocity_acceleration(lambda X,t:X+coefficient*t,X,.2357)
    torch.testing.assert_close(a,torch.zeros_like(X))
    _,v,a=compute_velocity_acceleration(lambda X,t:X+coefficient*torch.sin(2*t),X,.2357)
    torch.testing.assert_close(a,(-4*coefficient*torch.sin(X.new_tensor(2*.2357))).expand_as(X))


def test_hash_full_hessian_and_parameter_gradients():
    x=torch.tensor([[.2357,.387,.543]],dtype=torch.float64,requires_grad=True)
    embedding=torch.randn(35,2,dtype=x.dtype,requires_grad=True)
    offsets=torch.tensor([0,8,35])
    scale=torch.full((3,),2.,dtype=x.dtype)
    base=torch.full((3,),2.,dtype=x.dtype)
    function=lambda coordinates, table:interpolate_hash(coordinates,table,offsets,scale,base)
    assert torch.autograd.gradcheck(function,(x,embedding))
    assert torch.autograd.gradgradcheck(function,(x,embedding))
    y=function(x,embedding)
    first=torch.autograd.grad(y.sum(),x,create_graph=True)[0]
    second=torch.autograd.grad(first.sum(),x,create_graph=True)[0]
    second.square().sum().backward()
    assert embedding.grad is not None and torch.isfinite(embedding.grad).all()
