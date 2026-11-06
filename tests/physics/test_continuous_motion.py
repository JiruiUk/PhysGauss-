import torch
from models.continuous_motion import ContinuousSE3Motion,bspline_weights
from physics.kinematics import quaternion_matrix,compute_velocity_acceleration


def test_spline_partition_se3_and_analytic_derivatives():
    t=torch.linspace(0,1,17,dtype=torch.float64)
    weights=bspline_weights(t)
    torch.testing.assert_close(weights.sum(-1),torch.ones_like(t))
    model=ContinuousSE3Motion().double()
    model.field.bias.data.normal_(std=.01)
    X=torch.randn(4,3,dtype=t.dtype)
    def position(X,t):
        q,b=model(X,t)
        return (quaternion_matrix(q)@X[...,None]).squeeze(-1)+b
    q,b=model(X,torch.zeros(4,1,dtype=t.dtype))
    torch.testing.assert_close(q,torch.tensor([[1.,0,0,0]],dtype=t.dtype).expand(4,4))
    torch.testing.assert_close(b,torch.zeros_like(X))
    _,v,a=compute_velocity_acceleration(position,X,.2357)
    assert torch.isfinite(v).all() and torch.isfinite(a).all()
    a.square().mean().backward()
    assert torch.isfinite(model.field.weight.grad).all()
