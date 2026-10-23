"""Dimensionless isotropy penalty for Gaussian principal-axis scales."""
import math
import torch


def isotropic_normalization(scales, eps=1e-8):
    """Mean absolute deviation of |s| / mean(|s|) from one.

    PIDG adds signed deformation offsets to exp(canonical log-scales). The
    covariance uses squared axis lengths, so isotropy concerns their magnitudes.
    The epsilon floor protects the dimensionless mean for degenerate Gaussians.
    """
    if scales.ndim != 2 or scales.shape[1] != 3 or not scales.is_floating_point():
        raise ValueError('Expected floating-point scales with shape [N, 3]')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    if scales.shape[0] == 0:
        return scales.sum() * 0
    # Normalize twice to avoid overflow when axis lengths approach dtype limits.
    lengths = scales.abs()
    largest = lengths.amax(dim=-1, keepdim=True)
    largest = torch.where(largest > 0, largest, torch.ones_like(largest))
    relative = lengths / largest
    mean = relative.mean(dim=-1, keepdim=True).clamp_min(eps)
    # All-zero axes carry no shape information and receive zero penalty.
    normalized = relative / mean
    deviation = (normalized - 1).abs().mean(dim=-1)
    return torch.where(lengths.amax(dim=-1) > 0, deviation, torch.zeros_like(deviation)).mean()
