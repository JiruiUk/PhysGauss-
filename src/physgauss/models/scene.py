from torch import nn
from .gaussians import CanonicalGaussians
from .fields import DeformationField, MaterialField, UncertaintyField


class PhysGaussScene(nn.Module):
    def __init__(self, xyz, rgb, scales, config):
        super().__init__()
        self.canonical = CanonicalGaussians(xyz, rgb, scales)
        self.deformation = DeformationField(config.hidden_dim, config.frequencies, config.canonical_time)
        self.material = MaterialField(config.hidden_dim, config.frequencies)
        self.uncertainty = UncertaintyField(config.hidden_dim, config.frequencies)
        if config.freeze_canonical_geometry:
            for name in ("means", "log_scales", "quaternions"):
                getattr(self.canonical, name).requires_grad_(False)

    def forward(self, time):
        canonical = self.canonical()
        dynamic = self.deformation(canonical, time)
        variance = self.uncertainty(canonical.means, time)
        return dynamic, variance
