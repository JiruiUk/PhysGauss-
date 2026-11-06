import torch


def build_neighbor_graph(x, k=16):
    assert x.ndim == 2 and x.shape[1] == 3
    if len(x) < 1: raise ValueError("No particles")
    with torch.no_grad():
        # Includes the center particle as quadrature point; discrete selection is fixed in backward.
        return torch.cdist(x.detach(),x.detach()).topk(min(k,len(x)),largest=False).indices


def gaussian_test_weights(x, centers, bandwidth):
    delta = x[None] - centers[:,None]
    bandwidth = torch.as_tensor(bandwidth, device=x.device, dtype=x.dtype).reshape(-1,1,1)
    if (bandwidth.detach() <= 0).any(): raise ValueError("Positive bandwidth required")
    weights = torch.exp(-delta.square().sum(-1,keepdim=True)/(2*bandwidth.square()))
    gradient = -weights*delta/bandwidth.square()
    return weights[...,0], gradient


def weak_momentum_residual(x, rho, acceleration, stress, volumes, body_force=None,
                           k=16, bandwidth_scale=1.5, boundary_traction=None,
                           boundary_weights=None):
    n = len(x)
    assert x.shape == acceleration.shape == (n,3)
    assert rho.shape == volumes.shape == (n,1) and stress.shape == (n,3,3)
    graph = build_neighbor_graph(x,k)
    centers = x.detach()
    with torch.no_grad():
        distances = (x.detach()[graph]-centers[:,None]).norm(dim=-1)
        h = distances.max(-1).values.clamp_min(1e-3)*bandwidth_scale
    phi, grad = gaussian_test_weights(x,centers,h)
    mask = torch.zeros(n,n,device=x.device,dtype=x.dtype).scatter_(1,graph,1)
    phi, grad = phi*mask, grad*mask[...,None]
    force = torch.zeros_like(acceleration) if body_force is None else body_force
    inertia = torch.einsum("ji,ic,ia->jc",phi,rho*acceleration-force,volumes)
    internal = torch.einsum("jib,icb,ia->jc",grad,stress,volumes)
    residual = inertia+internal
    if boundary_traction is not None:
        if boundary_weights is None: raise ValueError("Boundary test weights/quadrature required")
        assert boundary_weights.shape[0] == n
        residual = residual - boundary_weights@boundary_traction
    # Per-test control-volume normalization keeps force-density units independent of N.
    measure = (phi*volumes[:,0]).sum(-1,keepdim=True).clamp_min(1e-12)
    return residual/measure
