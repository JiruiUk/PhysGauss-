import torch
from torch import nn
from torch.nn import functional as F
from .gaussians import GaussianState, quaternion_multiply


def expand_time(x, time):
    return torch.as_tensor(time, dtype=x.dtype, device=x.device).reshape(-1, 1).expand(len(x), 1)


class FourierMLP(nn.Module):
    """CPU reference encoder; replace behind the same interface with 4D hash grids."""
    def __init__(self, output_dim, hidden_dim=32, frequencies=2):
        super().__init__()
        self.register_buffer("bands", (2.0 ** torch.arange(frequencies)) * torch.pi)
        self.net = nn.Sequential(
            nn.Linear(4 * (1 + 2*frequencies), hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, time):
        xt = torch.cat([x, expand_time(x, time)], dim=-1)
        waves = xt[..., None] * self.bands
        return self.net(torch.cat([xt, waves.sin().flatten(1), waves.cos().flatten(1)], dim=-1))


class DeformationField(nn.Module):
    def __init__(self, hidden_dim=32, frequencies=2, canonical_time=0.0):
        super().__init__()
        self.field = FourierMLP(9, hidden_dim, frequencies)
        self.canonical_time = canonical_time
        nn.init.normal_(self.field.net[-1].weight, std=0.001)
        nn.init.zeros_(self.field.net[-1].bias)

    def displacement(self, x, time):
        dt = expand_time(x, time) - self.canonical_time
        return dt * self.field(x, time)[:, :3]

    def forward(self, state, time):
        raw = self.field(state.means, time)
        dt = expand_time(state.means, time) - self.canonical_time
        delta_q = torch.cat([torch.ones_like(dt), 0.5 * dt * raw[:, 6:9]], dim=-1)
        return GaussianState(
            state.means + dt * raw[:, :3],
            state.scales * (dt * raw[:, 3:6]).clamp(-2, 2).exp(),
            F.normalize(quaternion_multiply(delta_q, state.quaternions), dim=-1),
            state.colors, state.opacities,
        )


class MaterialField(nn.Module):
    """Eulerian velocity and symmetric Cauchy stress; density is a known input."""
    def __init__(self, hidden_dim=32, frequencies=2):
        super().__init__()
        self.field = FourierMLP(9, hidden_dim, frequencies)

    def forward(self, x, time):
        raw = self.field(x, time)
        xx, yy, zz, xy, xz, yz = raw[:, 3:].unbind(-1)
        stress = torch.stack([xx, xy, xz, xy, yy, yz, xz, yz, zz], -1).reshape(-1, 3, 3)
        return raw[:, :3], stress


class UncertaintyField(nn.Module):
    """Bounded photometric variance; this is not a calibrated physical posterior."""
    def __init__(self, hidden_dim=32, frequencies=2):
        super().__init__()
        self.field = FourierMLP(1, hidden_dim, frequencies)
        nn.init.zeros_(self.field.net[-1].weight)
        nn.init.constant_(self.field.net[-1].bias, -2.0)

    def forward(self, x, time):
        return 0.0001 + 0.0999 * self.field(x, time).sigmoid()
