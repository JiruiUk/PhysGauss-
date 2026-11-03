"""Differentiable local-affine support transport; no material law or force."""
import math
import torch
from ..invertible_motion.core import encode


def encoded_spatial_derivative(x):
    """Derivative of the archived coupling's two-band, two-coordinate encoding."""
    eye = torch.eye(2, device=x.device, dtype=x.dtype)
    frequency = x.new_tensor([math.pi, 2*math.pi])
    phase = x[..., None]*frequency
    sine = ((phase.cos()*frequency)[..., None]*eye[None, :, None, :]).flatten(1, 2)
    cosine = ((-phase.sin()*frequency)[..., None]*eye[None, :, None, :]).flatten(1, 2)
    return torch.cat([eye.expand(len(x), 2, 2), sine, cosine], 1)


def motion_with_differential(motion, x, t):
    """Exact chain rule for independent smooth I7 affine-coupling layers.

    All operations retain parameter, coordinate and time gradients, including
    the coordinate dependence of J. No detach, finite-difference approximation,
    neighborhood graph, inferred labels or relaxed-material claim.
    """
    if x.ndim != 2 or x.shape[-1] != 3 or t.shape != (len(x), 1):
        raise ValueError('Expected Nx3 coordinates and Nx1 times')
    identity = torch.eye(3, device=x.device, dtype=x.dtype).expand(len(x), 3, 3)
    jacobian = identity
    for layer in motion.layers:
        spatial = x[:, layer.other]
        encoded = torch.cat([encode(spatial, 2), encode(t, 4)], -1)
        first, second, last = layer.conditioner[0], layer.conditioner[2], layer.conditioner[4]
        h1 = torch.tanh(first(encoded))
        d1 = (1-h1.square())[:, :, None]*torch.einsum('oi,nij->noj', first.weight[:, :10], encoded_spatial_derivative(spatial))
        h2 = torch.tanh(second(h1))
        d2 = (1-h2.square())[:, :, None]*torch.einsum('oi,nij->noj', second.weight, d1)
        raw = last(h2)
        derivative = torch.einsum('oi,nij->noj', last.weight, d2)
        bounded = raw.tanh()
        scale = layer.log_scale_bound*bounded[:, :1]
        shift = layer.translation_bound*bounded[:, 1:]
        dscale = layer.log_scale_bound*(1-bounded[:, :1].square())*derivative[:, 0]
        dshift = layer.translation_bound*(1-bounded[:, 1:].square())*derivative[:, 1]
        diagonal = scale.exp()
        cross = x[:, layer.axis:layer.axis+1]*diagonal*dscale+dshift
        row = torch.stack([diagonal[:, 0] if axis == layer.axis else cross[:, layer.other.index(axis)]
                           for axis in range(3)], -1)
        local = torch.stack([row if axis == layer.axis else identity[:, axis] for axis in range(3)], 1)
        jacobian = local@jacobian
        changed = x[:, layer.axis:layer.axis+1]*diagonal+shift
        x = torch.cat([changed if axis == layer.axis else x[:, axis:axis+1] for axis in range(3)], -1)
    return x, jacobian


def covariance(scales, quaternion):
    """CUDA covariance convention; caller supplies unit canonical quaternions.

    Kept as the raw polynomial to match the archived static renderer as closely
    as floating point allows. Dynamic covariance has no independent warp head.
    """
    r, x, y, z = quaternion.unbind(-1)
    rotation = torch.stack([1-2*(y*y+z*z), 2*(x*y-r*z), 2*(x*z+r*y),
                            2*(x*y+r*z), 1-2*(x*x+z*z), 2*(y*z-r*x),
                            2*(x*z-r*y), 2*(y*z+r*x), 1-2*(x*x+y*y)], -1).reshape(-1, 3, 3)
    factor = rotation*scales[:, None, :]
    return factor@factor.transpose(-1, -2)


def transport(jacobian, reference):
    return jacobian@reference@jacobian.transpose(-1, -2)


def pack(matrix):
    return matrix[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]].contiguous()
