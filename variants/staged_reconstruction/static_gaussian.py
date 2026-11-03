"""An independent static parameter bank rendered by unchanged PIDG Gaussian code."""
import numpy as np
import torch
from torch import nn


class StaticGaussian(nn.Module):
    max_sh_degree = 3
    active_sh_degree = 3

    def __init__(self, initialization, device='cuda'):
        super().__init__()
        with np.load(initialization,allow_pickle=False) as data:
            xyz=torch.tensor(data['xyz'],device=device)
            rgb=torch.tensor(data['rgb'],device=device)
            self.spacing=float(data['spacing'])
        self.register_buffer('initial_xyz',xyz.clone())
        self._xyz=nn.Parameter(xyz)
        self._features_dc=nn.Parameter(((rgb-.5)/.28209479177387814)[:,None,:])
        self._features_rest=nn.Parameter(torch.zeros(len(xyz),15,3,device=device))
        self._opacity=nn.Parameter(torch.zeros(len(xyz),1,device=device))
        self._scaling=nn.Parameter(torch.full_like(xyz,np.log(self.spacing*.65)))
        rotation=torch.zeros(len(xyz),4,device=device);rotation[:,0]=1
        self._rotation=nn.Parameter(rotation)

    @property
    def get_xyz(self):return self._xyz
    @property
    def get_scaling(self):return self._scaling.exp()
    @property
    def get_rotation(self):return torch.nn.functional.normalize(self._rotation,dim=-1)
    @property
    def get_opacity(self):return self._opacity.sigmoid()
    @property
    def get_features(self):return torch.cat([self._features_dc,self._features_rest],dim=1)

    @torch.no_grad()
    def constrain(self):
        self._xyz.copy_(torch.maximum(torch.minimum(self._xyz,self.initial_xyz+self.spacing*2),self.initial_xyz-self.spacing*2))
        self._scaling.clamp_(np.log(self.spacing*.15),np.log(self.spacing*2.5))
