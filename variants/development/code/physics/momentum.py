import torch
from .differential import derivative


def strong_form_momentum_residual(rho, acceleration, stress, positions, body_force=None, F=None):
    """F supplied: positions are reference X; transform sigma derivatives to spatial x.

    d/dx_j = sum_K (F^-1)_{Kj} d/dX_K. Omitting F uses Eulerian coordinates.
    """
    n = len(positions)
    assert rho.shape == (n,1) and acceleration.shape == (n,3) and stress.shape == (n,3,3)
    grads = torch.stack([torch.stack([derivative(stress[:,i,j], positions) for j in range(3)],1)
                         for i in range(3)],1)
    if F is None:
        divergence = torch.einsum("nijk,jk->ni", grads, torch.eye(3,device=positions.device,dtype=positions.dtype))
    else:
        assert F.shape == (n,3,3)
        if (torch.linalg.det(F).detach() <= 0).any():
            raise ValueError("Strong momentum requires invertible positive deformation")
        divergence = torch.einsum("nijk,nkj->ni", grads, torch.linalg.inv(F))
    force = torch.zeros_like(acceleration) if body_force is None else body_force
    assert force.shape == acceleration.shape
    return rho*acceleration-divergence-force
