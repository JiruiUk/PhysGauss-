"""LPFM with frozen raster correspondence weights and exact Torch projections.

Projection and covariance gradients use Torch rather than the fork's metadata
backward, whose projected-mean gradient omits the pixel scaling factor.
"""
import math
import torch
from variants.kinematic_tuning.model import center_trajectory


def symmetric(v):
    a, b, c = v.unbind(-1)
    return torch.stack([a, b, b, c], -1).reshape(-1, 2, 2)


def sqrt_spd(matrix):
    eye = torch.eye(2, device=matrix.device, dtype=matrix.dtype)
    det = torch.linalg.det(matrix).clamp_min(1e-12).sqrt()
    denominator = (matrix.diagonal(dim1=-2, dim2=-1).sum(-1)+2*det).clamp_min(1e-12).sqrt()
    return (matrix+det[..., None, None]*eye)/denominator[..., None, None]


def rotation_raw(q):
    # Match CUDA computeCov3D: Gaussian base quaternion is normalized by its
    # getter; the additive deformation quaternion is not normalized again.
    r, x, y, z = q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-r*z), 2*(x*z+r*y),
                        2*(x*y+r*z), 1-2*(x*x+z*z), 2*(y*z-r*x),
                        2*(x*z-r*y), 2*(y*z+r*x), 1-2*(x*x+y*y)], -1).reshape(-1, 3, 3)


def projected(centers, scales, rotations, camera):
    view = camera.world_view_transform.to(centers)
    cam = centers@view[:3, :3]+view[3, :3]
    z = cam[:, 2].clamp_min(1e-6)
    tx, ty = math.tan(camera.FoVx/2), math.tan(camera.FoVy/2)
    fx, fy = camera.image_width/(2*tx), camera.image_height/(2*ty)
    # ndc2Pix maps the NDC centre to (width-1)/2, rather than width/2.
    mean = torch.stack([fx*cam[:, 0]/z+(camera.image_width-1)/2,
                        fy*cam[:, 1]/z+(camera.image_height-1)/2], -1)
    cx = (cam[:, 0]/z).clamp(-1.3*tx, 1.3*tx)*z
    cy = (cam[:, 1]/z).clamp(-1.3*ty, 1.3*ty)*z
    zero = torch.zeros_like(z)
    jac = torch.stack([fx/z, zero, -fx*cx/z.square(), zero, fy/z, -fy*cy/z.square()], -1).reshape(-1, 2, 3)
    rot = rotation_raw(rotations)
    covariance = (rot*scales[:, None, :].square())@rot.transpose(-1, -2)
    world_to_cam = view[:3, :3].T
    transform = jac@world_to_cam
    cov = transform@covariance@transform.transpose(-1, -2)
    cov = cov+.3*torch.eye(2, device=cov.device, dtype=cov.dtype)
    return mean, cov


def pixel_flow(package, next_mean, next_cov):
    ids = package['gs_per_pixel'].detach().long()
    safe = ids.clamp_min(0)
    valid = (ids >= 0) & (ids < len(next_mean))
    safe = safe.clamp_max(len(next_mean)-1)
    weights = package['weight_per_gs_pixel'].detach()*valid
    weights = weights/weights.sum(0, keepdim=True).clamp_min(1e-8)
    current_inverse = symmetric(package['conic_2D'].detach())
    # Culled Gaussians have zero covariance; they carry zero raster weight.
    eye = torch.eye(2, device=next_cov.device, dtype=next_cov.dtype)
    transform = sqrt_spd(next_cov)@sqrt_spd(current_inverse+1e-8*eye)
    shift = next_mean-package['proj_means_2D'].detach()
    offset = package['x_mu'].detach().permute(0, 2, 3, 1)
    local = torch.einsum('khwij,khwj->khwi', (transform-eye)[safe], offset)+shift[safe]
    return (local*weights[..., None]).sum(0), weights.sum(0)>0


def next_projection(gaussians, warp, camera, centers=None):
    if centers is None:
        centers = center_trajectory(gaussians.get_xyz, warp['d_xyz'])
    return projected(centers, gaussians.get_scaling+warp['d_scaling'], gaussians.get_rotation+warp['d_rotation'], camera)
