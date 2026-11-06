import torch
from torch import nn


class MaterialField(nn.Module):
    """Low-frequency canonical field. Time is intentionally absent."""
    def __init__(self, feature_dim=3, hidden=16):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(nn.Linear(3+feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].weight)
        self.net[-1].bias.data.copy_(torch.tensor([0.541323, .456758, -2.97063]))

    def forward(self, canonical_position, gaussian_feature=None):
        assert canonical_position.ndim == 2 and canonical_position.shape[1] == 3
        if gaussian_feature is None:
            gaussian_feature = canonical_position.new_zeros(len(canonical_position), self.feature_dim)
        assert gaussian_feature.shape == (len(canonical_position), self.feature_dim)
        raw = self.net(torch.cat([canonical_position, gaussian_feature], -1))
        return {"E": torch.nn.functional.softplus(raw[:, :1])+1e-6,
                "nu": .49*raw[:,1:2].sigmoid(), "viscosity": torch.nn.functional.softplus(raw[:,2:3])+1e-6,
                "features": raw}


def material_smoothness(x, params, k=8):
    if len(x) < 2:
        return params.sum()*0
    with torch.no_grad():
        distance = torch.cdist(x, x)
        distance.fill_diagonal_(float("inf"))
        neighbors = distance.topk(min(k,len(x)-1), largest=False).indices
        scale = distance.gather(1, neighbors).median().clamp_min(1e-6)
        weights = torch.exp(-distance.gather(1, neighbors).square()/(2*scale.square()))
    return (weights[...,None]*(params[:,None]-params[neighbors]).square()).mean()
