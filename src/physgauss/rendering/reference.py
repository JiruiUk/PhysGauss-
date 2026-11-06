from dataclasses import dataclass
from typing import Protocol
import torch
from torch import Tensor
from ..models.gaussians import GaussianState, quaternion_matrix


@dataclass
class Camera:
    K: Tensor
    world_to_camera: Tensor
    height: int
    width: int


@dataclass
class RenderResult:
    rgb: Tensor
    depth: Tensor
    alpha: Tensor
    variance: Tensor
    flow: Tensor | None = None


class Renderer(Protocol):
    def render(self, state: GaussianState, camera: Camera, variance: Tensor,
               next_state: GaussianState | None = None,
               next_camera: Camera | None = None) -> RenderResult: ...


def project_points(points, camera):
    p = points @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
    z = p[:, 2]
    uv = (p @ camera.K.T)[:, :2] / z[:, None].clamp_min(1e-4)
    return uv, z, p


class ReferenceRenderer:
    """Differentiable full-image splatter for tiny scenes: O(N*H*W), no CUDA extension."""
    def render(self, state, camera, variance, next_state=None, next_camera=None):
        if len(state.means)*camera.height*camera.width > 2_000_000:
            raise ValueError("Reference renderer is limited to N*H*W <= 2,000,000. Downsample for development "
                             "or integrate a tiled CUDA backend via the Renderer protocol.")
        uv, z, p = project_points(state.means, camera)
        rotation = quaternion_matrix(state.quaternions)
        cov_world = rotation @ torch.diag_embed(state.scales.square()) @ rotation.transpose(1, 2)
        rotation_camera = camera.world_to_camera[:3, :3]
        cov_camera = rotation_camera @ cov_world @ rotation_camera.T
        # Perspective Jacobian, assuming the validated zero-skew pinhole camera.
        safe_z = z.clamp_min(1e-4)
        zeros = torch.zeros_like(z)
        fx, fy = camera.K[0, 0], camera.K[1, 1]
        J = torch.stack([fx/safe_z, zeros, -fx*p[:, 0]/safe_z.square(),
                         zeros, fy/safe_z, -fy*p[:, 1]/safe_z.square()], -1).reshape(-1, 2, 3)
        cov_screen = J @ cov_camera @ J.transpose(1, 2)
        cov_screen = cov_screen + 0.25 * torch.eye(2, device=z.device, dtype=z.dtype)
        yy, xx = torch.meshgrid(
            torch.arange(camera.height, device=z.device, dtype=z.dtype),
            torch.arange(camera.width, device=z.device, dtype=z.dtype), indexing="ij",
        )
        pixels = torch.stack([xx, yy], -1)
        delta = pixels[None] - uv[:, None, None]
        exponent = torch.einsum("nhwi,nij,nhwj->nhw", delta, torch.linalg.inv(cov_screen), delta)
        visible = (z > 1e-3).to(z.dtype)
        alpha_i = (torch.exp(-0.5*exponent) * state.opacities[:, 0, None, None]
                   * visible[:, None, None]).clamp(max=0.99)
        order = z.argsort()
        alpha_i = alpha_i[order]
        transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha_i[:1]), 1-alpha_i], 0), 0)[:-1]
        weights = alpha_i * transmittance
        alpha = weights.sum(0)
        rgb = torch.einsum("nhw,nc->hwc", weights, state.colors[order])
        depth = torch.einsum("nhw,n->hw", weights, z[order]) / alpha.clamp_min(1e-6)
        # A per-pixel noise mixture, not propagation of independent Gaussian errors.
        pixel_variance = torch.einsum("nhw,n->hw", weights, variance[order, 0])
        pixel_variance = (pixel_variance + (1-alpha)*0.01).clamp_min(1e-4)
        flow = None
        if next_state is not None:
            next_uv, _, _ = project_points(next_state.means, next_camera or camera)
            particle_flow = next_uv - uv
            flow = torch.einsum("nhw,nc->hwc", weights, particle_flow[order]) / alpha[..., None].clamp_min(1e-6)
        return RenderResult(rgb, depth, alpha, pixel_variance, flow)
