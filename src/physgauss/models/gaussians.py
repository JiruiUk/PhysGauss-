from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class GaussianState:
    means: Tensor
    scales: Tensor
    quaternions: Tensor  # w, x, y, z
    colors: Tensor
    opacities: Tensor


def quaternion_matrix(q: Tensor) -> Tensor:
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y),
    ], dim=-1).reshape(-1, 3, 3)


def quaternion_multiply(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw-ax*bx-ay*by-az*bz,
        aw*bx+ax*bw+ay*bz-az*by,
        aw*by-ax*bz+ay*bw+az*bx,
        aw*bz+ax*by-ay*bx+az*bw,
    ], dim=-1)


class CanonicalGaussians(nn.Module):
    def __init__(self, xyz: Tensor, rgb: Tensor, scales: Tensor):
        super().__init__()
        self.means = nn.Parameter(xyz.clone())
        self.log_scales = nn.Parameter(scales.log().clone())
        q = torch.zeros(len(xyz), 4, device=xyz.device)
        q[:, 0] = 1
        self.quaternions = nn.Parameter(q)
        self.color_logits = nn.Parameter(torch.logit(rgb.clamp(0.001, 0.999)))
        self.opacity_logits = nn.Parameter(torch.full((len(xyz), 1), 1.0, device=xyz.device))

    def forward(self):
        return GaussianState(
            self.means, self.log_scales.clamp(-9, 2).exp(),
            F.normalize(self.quaternions, dim=-1),
            self.color_logits.sigmoid(), self.opacity_logits.sigmoid(),
        )
