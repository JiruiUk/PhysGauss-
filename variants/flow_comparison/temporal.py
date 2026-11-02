"""Differentiable material trajectories; no hash-encoder coordinate Hessians."""
from contextlib import contextmanager
import torch
from torch.utils.checkpoint import checkpoint
from variants.kinematic_tuning.model import center_trajectory


@contextmanager
def deterministic(deform):
    old = deform.reg_spatial_able, deform.reg_temporal_able
    deform.reg_spatial_able = deform.reg_temporal_able = False
    try:
        yield
    finally:
        deform.reg_spatial_able, deform.reg_temporal_able = old


def warp_at(deform, x, t):
    with deterministic(deform):
        return deform.step(x, x.new_full((len(x), 1), float(t)), fixed_attention=False)


def position(deform, x, t, chunk=4096):
    def evaluate(z):
        return center_trajectory(z, warp_at(deform, z, t)['d_xyz'])
    return torch.cat([checkpoint(evaluate, z, use_reentrant=False, preserve_rng_state=False)
                      if torch.is_grad_enabled() else evaluate(z) for z in x.split(chunk)])


def derivative(function, t, h, order=1):
    """Second-order centered/one-sided stencils on normalized time [0,1]."""
    if not 0 < h < .25 or not 0 <= t <= 1:
        raise ValueError('Invalid time/stencil')
    if order == 1:
        if t < h:
            return (-3*function(t)+4*function(t+h)-function(t+2*h))/(2*h)
        if t > 1-h:
            return (3*function(t)-4*function(t-h)+function(t-2*h))/(2*h)
        return (function(t+h)-function(t-h))/(2*h)
    if order == 2:
        if t < h:
            return (2*function(t)-5*function(t+h)+4*function(t+2*h)-function(t+3*h))/h**2
        if t > 1-h:
            return (2*function(t)-5*function(t-h)+4*function(t-2*h)-function(t-3*h))/h**2
        return (function(t+h)-2*function(t)+function(t-h))/h**2
    raise ValueError('Only velocity and acceleration are supported')


def velocity(gaussians, deform, x, ids, t, source, h=.001):
    if source == 'trajectory':
        return derivative(lambda s: position(deform, x, s), t, h)
    if source != 'independent':
        raise ValueError('Unknown velocity source')
    return gaussians.velocity_net(torch.cat([x, x.new_full((len(x), 1), t)], -1), ids)[0]
