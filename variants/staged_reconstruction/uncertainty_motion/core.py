"""Squared color-Jacobian support without the truncated PIDG export buffer.

For fixed geometry I = A c + b, independent pixel signs z give
E[(A.T z)**2] = diag(A.T A). RGB channels are independent probes of the
same scalar compositing operator. This estimates color support, not geometry
confidence, material identity, or physical correspondence.
"""
import torch


def color_vjp(rgb, colors, pixel_values):
    """Exact A.T pixel_values, keeping the diagnostic render graph reusable."""
    if rgb.ndim != 3 or colors.ndim != 2 or rgb.shape[0] != colors.shape[1]:
        raise ValueError('Expected CHW image and NC independent override colors')
    if pixel_values.shape != rgb.shape[1:]:
        raise ValueError('Pixel values must have shape HW')
    values = pixel_values.to(device=rgb.device, dtype=rgb.dtype)
    if not torch.isfinite(values).all():
        raise ValueError('Pixel values must be finite')
    gradient = torch.autograd.grad(rgb, colors, values.expand_as(rgb), retain_graph=True)[0]
    return gradient.detach().mean(dim=-1)


def squared_support(rgb, colors, *, batches, seed, pixel_weight=None):
    """Return a nonnegative estimate and sampling standard error per Gaussian.

    Uses a private generator; no global RNG or model .grad is modified. The
    caller must detach geometry and use independent override-color parameters.
    A pixel_weight w estimates sum_p w_p A_pi**2 (using sqrt(w) probes).
    Sampling SE is descriptive, not a calibrated uncertainty in 3D geometry.
    """
    if not isinstance(batches, int) or batches < 2:
        raise ValueError('At least two independent RGB probe batches required')
    if rgb.ndim != 3 or colors.ndim != 2 or rgb.shape[0] != colors.shape[1]:
        raise ValueError('Expected CHW image and NC independent override colors')
    if pixel_weight is None:
        scale = torch.ones_like(rgb[0])
    else:
        weight = pixel_weight.to(device=rgb.device, dtype=rgb.dtype)
        if weight.shape != rgb.shape[1:] or not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError('Pixel weights must be finite, nonnegative HW')
        scale = weight.sqrt()
    generator = torch.Generator(device=rgb.device).manual_seed(seed)
    total = torch.zeros(colors.shape[0], device=colors.device, dtype=torch.float64)
    total_square = torch.zeros_like(total)
    for _ in range(batches):
        signs = torch.randint(0, 2, rgb.shape, generator=generator, device=rgb.device, dtype=torch.int32)
        probe = (signs.to(rgb.dtype)*2-1)*scale
        gradient = torch.autograd.grad(rgb, colors, probe, retain_graph=True)[0].detach().double()
        square = gradient.square()
        total += square.sum(dim=-1)
        total_square += square.square().sum(dim=-1)
    samples = batches*colors.shape[1]
    mean = total/samples
    variance = ((total_square-samples*mean.square())/(samples-1)).clamp_min(0)
    if not torch.isfinite(mean).all():
        raise RuntimeError('Nonfinite color-Jacobian support')
    return {'support': mean, 'sampling_se': (variance/samples).sqrt(), 'samples': samples}


def recurring_candidates(support, qualified, *, fraction=.02, minimum_frames=5):
    """Rank only observation-qualified positive supports; never fill absent data.

    This count is prior to spatial sampling. Stable row IDs carry no certified
    physical identity. Ties are resolved by row ID for reproducibility.
    """
    if support.ndim != 2 or qualified.shape != support.shape or qualified.dtype != torch.bool:
        raise ValueError('Expected TN support and boolean qualification')
    if not 0 < fraction <= 1 or not isinstance(minimum_frames, int) or minimum_frames < 1:
        raise ValueError('Invalid recurring-selection settings')
    if not torch.isfinite(support).all() or (support < 0).any():
        raise ValueError('Support must be finite and nonnegative')
    count = max(1, int(support.shape[1]*fraction))
    masked = torch.where(qualified & (support > 0), support, torch.zeros_like(support))
    order = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :count]
    selected = torch.zeros_like(qualified)
    selected.scatter_(1, order, masked.gather(1, order) > 0)
    period = selected.sum(dim=0)
    return {'per_frame': selected, 'period': period, 'candidates': period >= minimum_frames}
