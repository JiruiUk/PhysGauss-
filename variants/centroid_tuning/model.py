"""Differentiable 2D alpha centroid; no claim of physical mass recovery."""
import math
import torch


def foreground_centroid(alpha, eps=1e-6):
    if alpha.ndim != 3 or alpha.shape[0] != 1 or min(alpha.shape[1:]) < 2:
        raise ValueError('Expected alpha shape [1,H,W] with H,W >= 2')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('Centroid epsilon must be positive and finite')
    if not torch.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError('Alpha must be finite and in [0,1]')
    height, width = alpha.shape[1:]
    x = torch.linspace(0, 1, width, device=alpha.device, dtype=alpha.dtype)
    y = torch.linspace(0, 1, height, device=alpha.device, dtype=alpha.dtype)
    image = alpha[0]
    # Average mass makes the denominator floor independent of image resolution.
    mass = image.mean().clamp_min(eps)
    first_x = (image.mean(0) * x).mean()
    first_y = (image.mean(1) * y).mean()
    return torch.stack((first_x, first_y)) / mass


def centroid_loss(alpha, target, eps=1e-6):
    if target.shape != (2,) or not torch.isfinite(target).all() or (target < 0).any() or (target > 1).any():
        raise ValueError('Target must be a finite normalized [x,y] centroid')
    return (foreground_centroid(alpha, eps) - target.detach().to(alpha)).square().sum()
