"""Pointwise relative log-stretch prior; no graph or material stress law."""
import torch
from ..covariance_transport.core import motion_with_differential


def relative_differential(current, reference):
    """J_t J_ref^-1 with gradients into both configurations, without an inverse."""
    if current.shape != reference.shape or current.ndim != 3 or current.shape[-2:] != (3, 3):
        raise ValueError('Expected matching Nx3x3 differentials')
    return torch.linalg.solve(reference.transpose(-1, -2), current.transpose(-1, -2)).transpose(-1, -2)


def log_stretch_energy(relative, robust_scale=0.3):
    """Nerfies log-singular-value energy with Geman-McClure remapping.

    The relative forward map is an adaptation. Orientation preservation comes
    from the inherited invertible architecture, not this singular-value loss.
    This is not a constitutive stress or a calibrated material energy.
    """
    if robust_scale <= 0 or relative.ndim != 3 or relative.shape[-2:] != (3, 3):
        raise ValueError('Expected Nx3x3 differentials and positive robust scale')
    singular = torch.linalg.svdvals(relative)
    if not bool(torch.isfinite(singular).all() and (singular > 0).all()):
        raise FloatingPointError('Non-finite or singular relative differential')
    energy = singular.log().square().sum(-1)
    robust = 2*energy/(energy+4*robust_scale**2)
    return robust, energy, singular


def sample_indices(count, samples, step, device):
    """Deterministic coverage without advancing image/physics sampler RNG."""
    if min(count, samples, step) < 1:
        raise ValueError('Positive count, sample budget and one-based step required')
    samples = min(samples, count)
    return (torch.arange(samples, device=device)+(step-1)*samples) % count


def strain_prior(motion, points, observed_time, step, plan):
    indices = sample_indices(len(points), plan['strain_samples'], step, points.device)
    query = points[indices]
    reference_time = plan['strain_reference_time']
    auxiliary_time = ((step-1) % 200)/199
    def differential(time):
        return motion_with_differential(motion, query, query.new_full((len(query), 1), float(time)))[1]
    reference = differential(reference_time)
    current = torch.cat([differential(observed_time), differential(auxiliary_time)])
    relative = relative_differential(current, reference.repeat(2, 1, 1))
    robust, energy, singular = log_stretch_energy(relative, plan['strain_robust_scale'])
    loss = robust.mean()
    return loss, {'raw_log_stretch_energy': float(energy.detach().mean()),
                  'min_stretch': float(singular.detach().min()),
                  'max_stretch': float(singular.detach().max()),
                  'sample_count': len(query), 'observed_time': float(observed_time),
                  'auxiliary_time': auxiliary_time, 'reference_time': reference_time}
