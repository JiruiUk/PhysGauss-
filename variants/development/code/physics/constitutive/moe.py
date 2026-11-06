import torch
from torch import nn
from .neo_hookean import NeoHookeanMaterial


class ConstitutiveMoE(nn.Module):
    """First practical mixture: elastic and near-rigid; gate is canonical, never time."""
    def __init__(self, feature_dim=3, rigid_multiplier=10.):
        super().__init__()
        self.gate = nn.Linear(feature_dim, 2)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.elastic = NeoHookeanMaterial()
        self.rigid_multiplier = rigid_multiplier

    def forward(self, F, F_dot, material_features, E=None, nu=None):
        assert material_features.shape == (len(F), self.gate.in_features)
        weights = self.gate(material_features).softmax(-1)
        sigma = self.elastic(F,E,nu)
        experts = torch.stack([sigma, self.rigid_multiplier*sigma],1)
        return (weights[...,None,None]*experts).sum(1), weights, experts
