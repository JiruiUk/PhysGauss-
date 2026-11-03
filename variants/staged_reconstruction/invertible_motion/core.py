"""Conditional affine coupling, independently tested before scene integration."""
import math
import torch
from torch import nn


def encode(value, bands):
    frequencies=value.new_tensor([math.pi*2**i for i in range(bands)])
    phase=value[...,None]*frequencies
    return torch.cat([value,phase.sin().flatten(-2),phase.cos().flatten(-2)],-1)


class Coupling(nn.Module):
    def __init__(self,axis,width=64,log_scale_bound=.2,translation_bound=.2):
        super().__init__()
        if axis not in [0,1,2] or width<1 or not 0<log_scale_bound<=1 or translation_bound<=0:
            raise ValueError('Invalid coupling capacity or bounds')
        self.axis=axis; self.other=[j for j in range(3) if j!=axis]
        self.log_scale_bound=log_scale_bound; self.translation_bound=translation_bound
        self.conditioner=nn.Sequential(nn.Linear(19,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,2))
        nn.init.zeros_(self.conditioner[-1].weight); nn.init.zeros_(self.conditioner[-1].bias)

    def coefficients(self,x,t):
        raw=self.conditioner(torch.cat([encode(x[:,self.other],2),encode(t,4)],-1))
        return self.log_scale_bound*raw[:,:1].tanh(), self.translation_bound*raw[:,1:].tanh()

    def forward(self,x,t,inverse=False):
        scale,shift=self.coefficients(x,t)
        changed=(x[:,self.axis:self.axis+1]-shift)*(-scale).exp() if inverse else x[:,self.axis:self.axis+1]*scale.exp()+shift
        output=torch.cat([changed if j==self.axis else x[:,j:j+1] for j in range(3)],-1)
        return output, -scale.squeeze(-1) if inverse else scale.squeeze(-1)


class InvertibleMotion(nn.Module):
    def __init__(self,width=64,log_scale_bound=.2,translation_bound=.2):
        super().__init__()
        self.layers=nn.ModuleList([Coupling(axis,width,log_scale_bound,translation_bound) for axis in [0,1,2,0,1,2]])

    def forward(self,x,t,inverse=False,return_logdet=False):
        if x.ndim!=2 or x.shape[-1]!=3 or t.shape!=(len(x),1):
            raise ValueError('Expected Nx3 coordinates and Nx1 times')
        logdet=x.new_zeros(len(x))
        for layer in reversed(self.layers) if inverse else self.layers:
            x,increment=layer(x,t,inverse); logdet=logdet+increment
        return (x,logdet) if return_logdet else x


class InvertibleDeform:
    """Preserve PIDG covariance heads, share one invertible position map."""
    def __init__(self,original,motion):
        self.original=original; self.motion=motion

    @property
    def reg_spatial_able(self): return self.original.reg_spatial_able

    @reg_spatial_able.setter
    def reg_spatial_able(self,value): self.original.reg_spatial_able=value

    @property
    def reg_temporal_able(self): return self.original.reg_temporal_able

    @reg_temporal_able.setter
    def reg_temporal_able(self,value): self.original.reg_temporal_able=value

    def step(self,xyz,t,fixed_attention=False):
        state=self.original.step(xyz,t,fixed_attention=fixed_attention)
        moved=self.motion(xyz,t)
        quaternion=xyz.new_zeros(len(xyz),4); quaternion[:,0]=1
        return dict(state,d_xyz=torch.cat([quaternion,moved-xyz],-1))


def install(scene,plan,enabled=False,optimizer=None):
    if not enabled or not plan.get('invertible_motion_enabled',False): return None
    if isinstance(scene.deforms[0],InvertibleDeform): raise ValueError('Already installed')
    # All random initialization is on CPU inside a fork, preserving the original
    # training RNG streams and the original sampler/physics graph state.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(plan['initialization_seed'])
        motion=InvertibleMotion(plan['width'],plan['log_scale_bound'],plan['translation_bound'])
    motion=motion.to(scene.centers.device)
    scene.networks.append(motion)
    adapter=InvertibleDeform(scene.deforms[0],motion); scene.deforms[0]=adapter
    if optimizer is not None:
        optimizer.add_param_group({'params':motion.parameters(),'lr':plan['learning_rate'],'name':'invertible_motion'})
    return adapter
