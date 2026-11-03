"""Native-renderer Gaussian nuisance planes and premultiplied composition."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn


def compose(static_rgb, nuisance_rgb=None, nuisance_alpha=None, enabled=False):
    if not enabled: return static_rgb
    assert nuisance_rgb is not None and nuisance_alpha is not None
    return nuisance_rgb+(1-nuisance_alpha)*static_rgb


class NuisancePlane(nn.Module):
    """A fixed-budget view-specific Gaussian plane, not a dynamic 3D object.

    World coordinates and two in-plane scales are differentiable. The plane's
    depth/orientation are fixed to avoid unidentifiable single-view geometry.
    Private NumPy RNG preserves the shared static trainer's sampler/RNG state.
    """
    active_sh_degree = 0
    max_sh_degree = 0

    def __init__(self, view, k, image, count=1000, seed=20261103, depth=.3, device='cuda'):
        super().__init__()
        height,width = image.shape[:2]; inverse = np.linalg.inv(view)
        half = depth*np.array([width/(2*k[0,0]),height/(2*k[1,1])])
        uv = np.random.default_rng(seed).uniform(-1,1,(count,2)).astype(np.float32)
        xy = np.rint((uv+1)*np.array([width-1,height-1])/2).astype(int)
        colors = image[xy[:,1],xy[:,0]]
        tensor = lambda x:torch.tensor(x,dtype=torch.float32,device=device)
        self._uv = nn.Parameter(tensor(uv)); self._rgb = nn.Parameter(tensor(colors))
        self._log_scale = nn.Parameter(tensor(np.broadcast_to(np.log(half*1.4/np.sqrt(count)),(count,2)).copy()))
        self._opacity = nn.Parameter(torch.full((count,1),float(np.log(.1/.9)),device=device))
        self.register_buffer('origin',tensor(inverse[:3,3])); self.register_buffer('basis',tensor(inverse[:3,:3]))
        self.register_buffer('plane_extent',tensor(half)); self.register_buffer('depth',tensor(depth))
        quaternion = Rotation.from_matrix(inverse[:3,:3]).as_quat()[[3,0,1,2]]
        self.register_buffer('rotation',tensor(np.broadcast_to(quaternion,(count,4)).copy()))
        self.register_buffer('thickness',torch.full((count,1),depth*.001,device=device))

    @property
    def get_xyz(self):
        local = torch.cat([self._uv*self.plane_extent,self.depth.expand(len(self._uv),1)],1)
        return (local@self.basis.T+self.origin).contiguous()

    @property
    def get_scaling(self):return torch.cat([self._log_scale.exp(),self.thickness],1)
    @property
    def get_rotation(self):return self.rotation
    @property
    def get_opacity(self):return self._opacity.sigmoid()
    @property
    def get_features(self):return ((self._rgb.clamp(0,1)-.5)/.28209479177387814)[:,None]
    @property
    def get_rgb(self):return self._rgb.clamp(0,1)

    @torch.no_grad()
    def constrain(self):
        self._uv.clamp_(-1.2,1.2); self._rgb.clamp_(0,1); self._opacity.clamp_(-12,8)
        self._log_scale.copy_(torch.maximum(torch.minimum(self._log_scale,(self.plane_extent*.4).log()),(self.plane_extent/400).log()))


def optimizer_for_nuisance(model):
    rates = {'_uv':.0016,'_rgb':.025,'_log_scale':.05,'_opacity':.05}
    return torch.optim.Adam([{'params':[getattr(model,n)],'lr':v,'name':n} for n,v in rates.items()],eps=1e-15)


def support_penalty(static_alpha,reliable,nuisance_alpha,static_weight=.01,nuisance_weight=.01):
    # Unknown pixels never become an alpha-one static prior.
    weight = reliable.to(static_alpha.dtype)
    static = ((1-static_alpha.squeeze(0)).abs()*weight).sum()/weight.sum().clamp_min(1)
    nuisance = nuisance_alpha.mean() if nuisance_alpha is not None else static_alpha.sum()*0
    return static_weight*static+nuisance_weight*nuisance
