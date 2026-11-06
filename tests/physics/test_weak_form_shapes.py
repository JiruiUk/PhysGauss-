import torch
from physics.weak_form import build_neighbor_graph,gaussian_test_weights,weak_momentum_residual


def test_gaussian_gradients_and_manufactured_uniform_acceleration():
    x=torch.randn(12,3,dtype=torch.float64,requires_grad=True)
    centers=x.detach()[:3]
    phi,grad=gaussian_test_weights(x,centers,.8)
    for row in range(3):
        expected=torch.autograd.grad(phi[row].sum(),x,retain_graph=True)[0]
        torch.testing.assert_close(grad[row],expected)
    assert build_neighbor_graph(x,16).shape==(12,12)
    rho=torch.ones(12,1,dtype=x.dtype)
    a=torch.ones_like(x)*.2
    stress=x[:,:,None]*x[:,None,:]*0
    volume=rho/12
    residual=weak_momentum_residual(x,rho,a,stress,volume)
    torch.testing.assert_close(residual,a)
    zero=weak_momentum_residual(x,rho,a,stress,volume,body_force=rho*a)
    torch.testing.assert_close(zero,x*0)
    assert residual.shape==(12,3)
