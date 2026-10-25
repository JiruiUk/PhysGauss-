"""Full Gaussian-center kinematics in units of scene distance / normalized time."""
import math
import torch
import torch.nn.functional as F


def center_trajectory(x, warp):
    # Exactly the renderer's normalized-quaternion rotation, plus translation.
    q = warp[..., :4]
    r, i, j, k = q.unbind(-1)
    two_s = 2 / q.square().sum(-1)
    rotation = torch.stack((1-two_s*(j*j+k*k), two_s*(i*j-k*r), two_s*(i*k+j*r),
                            two_s*(i*j+k*r), 1-two_s*(i*i+k*k), two_s*(j*k-i*r),
                            two_s*(i*k-j*r), two_s*(j*k+i*r), 1-two_s*(i*i+j*j)), -1)
    return (rotation.reshape(*q.shape[:-1], 3, 3) @ x.unsqueeze(-1)).squeeze(-1) + warp[..., 4:]


def trajectory_velocity(trajectory, x, t, h=1e-3):
    """Second-order finite differences, with one-sided boundary stencils.

    Backpropagates into both stencil evaluations. This avoids requiring the
    custom CUDA hash encoder's unsupported coordinate second derivatives.
    h is in normalized scene time, not seconds; no division by frame count.
    """
    if not math.isfinite(h) or not 0 < h < .25:
        raise ValueError('Finite difference step must lie in (0, .25)')
    if not torch.isfinite(t).all() or not ((t >= 0) & (t <= 1)).all():
        raise ValueError('Time must lie in [0, 1]')
    lower, upper = t < h, t > 1-h
    t0 = t
    t1 = torch.where(upper, t-h, torch.where(lower, t+h, t-h))
    t2 = torch.where(upper, t-2*h, torch.where(lower, t+2*h, t+h))
    c0 = torch.where(lower, -3., torch.where(upper, 3., 0.))
    c1 = torch.where(lower, 4., torch.where(upper, -4., -1.))
    c2 = torch.where(lower, -1., torch.where(upper, 1., 1.))
    return (c0*trajectory(x, t0) + c1*trajectory(x, t1) + c2*trajectory(x, t2))/(2*h)


def consistency_loss(predicted, derived, delta=.1):
    return F.huber_loss(predicted, derived, delta=delta)
