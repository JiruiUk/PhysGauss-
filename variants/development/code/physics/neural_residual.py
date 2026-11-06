import torch
from torch import nn


class BoundedStressResidual(nn.Module):
    def __init__(self, epsilon=.01, max_ratio=.2):
        super().__init__()
        if epsilon <= 0 or not 0 < max_ratio <= .2: raise ValueError("Invalid stress residual bound")
        self.epsilon, self.max_ratio = epsilon, max_ratio
        self.net = nn.Sequential(nn.Linear(12,16),nn.Tanh(),nn.Linear(16,9))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, stress, F, canonical_position):
        raw = self.net(torch.cat([F.flatten(1),canonical_position],-1)).reshape(-1,3,3)
        raw = .5*(raw+raw.transpose(-1,-2))
        proposed = self.epsilon*raw
        norm = (proposed.square().sum(dim=(-2,-1),keepdim=True)+1e-24).sqrt()
        cap = self.max_ratio*((stress.square().sum(dim=(-2,-1),keepdim=True)+1e-12).sqrt()-1e-6)
        correction = proposed*torch.minimum(torch.ones_like(cap),cap/norm)
        ratio = correction.norm(dim=(-2,-1))/(stress.norm(dim=(-2,-1))+1e-12)
        return stress+correction, raw.square().mean(), ratio
