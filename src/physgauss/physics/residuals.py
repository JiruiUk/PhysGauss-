import torch
from ..models.fields import expand_time


def derivative(y, x):
    """Per-sample derivative; fields must not mix samples (no batch normalization)."""
    if not y.requires_grad:
        return torch.zeros_like(x)
    grad = torch.autograd.grad(y.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(x) if grad is None else grad


def momentum_residual(material, positions, time, density=1.0, body_acceleration=None):
    # Independent Eulerian coordinates; PDE regularizes the material field rather than
    # backpropagating through the choice of collocation locations.
    x = positions.detach().clone().requires_grad_(True)
    t = expand_time(x, time).detach().clone().requires_grad_(True)
    velocity, stress = material(x, t)
    spatial = torch.stack([derivative(velocity[:, i], x) for i in range(3)], dim=1)
    temporal = torch.cat([derivative(velocity[:, i], t) for i in range(3)], dim=1)
    convective = torch.einsum("nij,nj->ni", spatial, velocity)
    divergence = torch.stack([
        sum(derivative(stress[:, i, j], x)[:, j] for j in range(3)) for i in range(3)
    ], dim=1)
    acceleration = x.new_tensor(body_acceleration or [0, 0, 0])
    return density * (temporal + convective - acceleration) - divergence


def trajectory_residual(deformation, material, canonical_positions, time):
    t = expand_time(canonical_positions, time).detach().clone().requires_grad_(True)
    positions = canonical_positions + deformation.displacement(canonical_positions, t)
    trajectory_velocity = torch.cat([derivative(positions[:, i], t) for i in range(3)], dim=1)
    predicted_velocity, _ = material(positions, t)
    return trajectory_velocity - predicted_velocity
