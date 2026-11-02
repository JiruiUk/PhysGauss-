"""Affine projected flow matching the actual CUDA metadata conventions.

CUDA stores x_mu = projected_mean - pixel and records alpha*T AFTER updating
T, unlike RGB compositing. Reconstruct pre-update compositing weights from
the actual Gaussian opacity/conic, and use the negative metadata offset.
Visibility and current correspondences are frozen, as in the original arms.
"""
import torch
from variants.flow_comparison.flow import symmetric, sqrt_spd, next_projection, projected


def contribution_weights(package, opacities):
    ids = package['gs_per_pixel'].detach().long()
    valid = (ids >= 0) & (ids < len(opacities))
    safe = ids.clamp(0, len(opacities)-1)
    with torch.no_grad():
        inverse = symmetric(package['conic_2D'].detach())[safe]
        offset = package['x_mu'].detach().permute(0, 2, 3, 1)
        power = -.5*torch.einsum('khwi,khwij,khwj->khw', offset, inverse, offset)
        alpha = (opacities.detach().reshape(-1)[safe]*power.exp()).clamp_max(.99)
        alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
        transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[:1]), 1-alpha[:-1]], 0), 0)
        weights = alpha*transmittance
    return safe, weights


def pixel_flow(package, next_mean, next_cov, opacities):
    safe, weights = contribution_weights(package, opacities)
    weights = weights/weights.sum(0, keepdim=True).clamp_min(1e-8)
    current_inverse = symmetric(package['conic_2D'].detach())
    eye = torch.eye(2, device=next_cov.device, dtype=next_cov.dtype)
    transform = sqrt_spd(next_cov)@sqrt_spd(current_inverse+1e-8*eye)
    shift = next_mean-package['proj_means_2D'].detach()
    # Material offset is pixel-mean; CUDA metadata stores its negative.
    offset = -package['x_mu'].detach().permute(0, 2, 3, 1)
    local = torch.einsum('khwij,khwj->khwi', (transform-eye)[safe], offset)+shift[safe]
    return (local*weights[..., None]).sum(0), weights.sum(0)>0
