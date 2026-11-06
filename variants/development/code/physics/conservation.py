import torch


def density_from_jacobian(rho0, J):
    assert J.ndim == 2 and J.shape[1] == 1
    if (J.detach() <= 0).any():
        raise ValueError("Density requires positive Jacobians; do not silently clamp inverted particles")
    return torch.as_tensor(rho0, device=J.device, dtype=J.dtype) / J


def mass_conservation_loss(rho, J, rho0):
    assert rho.shape == J.shape and rho.ndim == 2 and rho.shape[1] == 1
    return ((rho * J - rho0) / torch.as_tensor(rho0, device=J.device).clamp_min(1e-12)).square().mean()


def volume_preservation_loss(J):
    assert J.ndim == 2 and J.shape[1] == 1
    return (J-1).square().mean()
