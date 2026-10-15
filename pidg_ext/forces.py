"""Low-dimensional shared body-force density; no per-Gaussian force parameters."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]


def _path(value):
    path=Path(value)
    return path if path.is_absolute() else ROOT/path


def interpolate_controls(controls,t):
    u=t.clamp(0,1)*(len(controls)-1)
    left=u.floor().long().clamp(0,len(controls)-2)
    alpha=(u-left).reshape(len(t),*([1]*(controls.ndim-1)))
    return controls[left]*(1-alpha)+controls[left+1]*alpha


class StructuredForce(nn.Module):
    def __init__(self,config):
        super().__init__();self.mode=config.get('force_mode','low_rank')
        self.input_hashes={str(_path(config[key]).resolve()):hashlib.sha256(_path(config[key]).read_bytes()).hexdigest()
                           for key in ('force_surface_file','known_force_file') if config.get(key)}
        self.coefficients=None
        controls=config.get('force_controls',4)
        if controls<2 or controls>8:raise ValueError('Use 2-8 shared temporal controls')
        if self.mode=='low_rank':
            self.rank=config.get('force_rank',3)
            if not 1<=self.rank<=8:raise ValueError('Force rank must be 1-8')
            self.coefficients=nn.Parameter(torch.zeros(controls,self.rank,3,dtype=torch.float64))
        elif self.mode=='surface_normal':
            data=np.load(_path(config['force_surface_file']),allow_pickle=False)
            normals=torch.as_tensor(data['normals'],dtype=torch.float64)
            mask=torch.as_tensor(data['mask'],dtype=torch.bool)
            if normals.shape!=(len(mask),3) or not torch.isfinite(normals).all() or (normals[mask].norm(dim=-1)<=1e-8).any():
                raise ValueError('Invalid supplied surface mask/normals')
            self.register_buffer('normals',normals/normals.norm(dim=-1,keepdim=True).clamp_min(1e-12))
            self.register_buffer('mask',mask)
            self.coefficients=nn.Parameter(torch.zeros(controls,dtype=torch.float64))
        elif self.mode=='known':
            if config.get('known_force_file'):
                data=np.load(_path(config['known_force_file']),allow_pickle=False)
                times=torch.as_tensor(data['times'],dtype=torch.float64)
                forces=torch.as_tensor(data['force'],dtype=torch.float64)
                if len(times)<2 or forces.shape[0]!=len(times) or forces.ndim!=3 or forces.shape[-1]!=3 or not (times.diff()>0).all() or not torch.isfinite(forces).all():
                    raise ValueError('Invalid known force samples')
                self.register_buffer('known_times',times);self.register_buffer('known_values',forces)
            else:
                value=torch.as_tensor(config['known_force'],dtype=torch.float64)
                if value.shape!=(3,) or not torch.isfinite(value).all():raise ValueError('Known force must be a finite vector')
                self.register_buffer('known_constant',value)
        elif self.mode!='none':raise ValueError('Unknown force mode')

    def forward(self,X,t,particle_ids=None,coefficients=None):
        t=torch.as_tensor(t,device=X.device,dtype=X.dtype).reshape(-1)
        if len(t)==1:t=t.expand(len(X))
        if len(t)!=len(X):raise ValueError('Time must be scalar or one value per query')
        if self.mode=='none':return torch.zeros_like(X)
        if self.mode=='known':
            if hasattr(self,'known_constant'):return self.known_constant.to(X).expand_as(X)
            times=self.known_times.to(X)
            if t.min()<times[0]-1e-6 or t.max()>times[-1]+1e-6:raise ValueError('Known force outside sampled time range')
            if particle_ids is None:raise ValueError('Known sampled force needs particle IDs')
            ids=particle_ids.to(X.device)
            if ids.min()<0 or ids.max()>=self.known_values.shape[1]:raise ValueError('Unknown force particle ID')
            left=(torch.searchsorted(times,t.contiguous(),right=True)-1).clamp(0,len(times)-2)
            alpha=((t-times[left])/(times[left+1]-times[left]))[:,None]
            values=self.known_values.to(X)
            return values[left,ids]*(1-alpha)+values[left+1,ids]*alpha
        coefficients=(self.coefficients if coefficients is None else coefficients).to(X)
        temporal=interpolate_controls(coefficients,t)
        if self.mode=='surface_normal':
            if particle_ids is None:raise ValueError('Surface force needs supplied particle IDs')
            ids=particle_ids.to(X.device)
            if ids.min()<0 or ids.max()>=len(self.mask):raise ValueError('No surface annotation for particle')
            return self.normals.to(X)[ids]*self.mask.to(X.device)[ids,None]*temporal[:,None]
        x,y,z=X.unbind(-1)
        basis=torch.stack((torch.ones_like(x),x,y,z,x*x,y*y,z*z,x*y),dim=-1)[:,:self.rank]
        return (basis[...,None]*temporal).sum(1)

    def regularizers(self,X,t,particle_ids=None):
        force=self(X,t,particle_ids)
        return {'magnitude':force.square().sum(-1).mean(),
                'temporal':self.coefficients.diff(dim=0).square().mean().to(X) if self.coefficients is not None else X.new_zeros(()),
                'sparsity':force.abs().sum(-1).mean()}


def momentum_components(coords,velocity_net,force_sigma,rho=1.):
    # Retain the released PIDG convective contraction and stress-divergence code.
    from motion_utils.pidg import compute_acceleration,compute_stress_divergence
    velocity,stress=velocity_net(coords)
    dvdt=compute_acceleration(velocity_net,coords,1e-3)
    grad_v=torch.stack([torch.autograd.grad(velocity[:,i].sum(),coords,create_graph=True,retain_graph=True)[0][:,:3] for i in range(3)],dim=1)
    inertial=rho*(dvdt+torch.einsum('bi,bij->bj',velocity,grad_v))
    divergence=compute_stress_divergence(coords,velocity,stress,force_sigma)
    return inertial,divergence
