"""MovingParts-inspired shared motion with a retained nonrigid C1 residual.

Candidate capacity is not an object-count label. No instance masks, pseudo IDs,
material assignments, free stiffness or forces are used by this module.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def rotation(q):
    q=F.normalize(q,dim=-1,eps=1e-8);w,x,y,z=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


def encode(x,bands=4):
    phase=x[...,None]*(2**torch.arange(bands,device=x.device,dtype=x.dtype))*math.pi
    return torch.cat((x,phase.sin().flatten(-2),phase.cos().flatten(-2)),-1)


class ObjectMotion(nn.Module):
    def __init__(self,capacity=16,learn_identity=True):
        super().__init__()
        if capacity<2:raise ValueError('At least two candidate slots required')
        self.capacity=capacity
        self.identity=nn.Sequential(nn.Linear(27,64),nn.SiLU(),nn.Linear(64,capacity))
        self.motion=nn.Sequential(nn.Linear(9,64),nn.SiLU(),nn.Linear(64,capacity*7))
        nn.init.zeros_(self.identity[-1].weight);nn.init.zeros_(self.identity[-1].bias)
        nn.init.zeros_(self.motion[-1].weight);nn.init.zeros_(self.motion[-1].bias)
        self.identity.requires_grad_(learn_identity)
        self.register_buffer('anchors',torch.zeros(capacity,3))
        self.register_buffer('origin',torch.zeros(3));self.register_buffer('scale',torch.ones(3))
        self.register_buffer('radius',torch.tensor(1.))
        self.register_buffer('active',torch.ones(capacity,dtype=torch.bool))
        self.register_buffer('initialized',torch.tensor(False))

    @torch.no_grad()
    def initialize(self,x):
        if bool(self.initialized):return
        if x.ndim!=2 or x.shape[1]!=3 or len(x)<self.capacity or not torch.isfinite(x).all():
            raise ValueError('Finite canonical points and sufficient candidates required')
        # Deterministic FPS is geometric initialization, not an instance target.
        sample=x[::max(1,len(x)//8192)][:8192]
        self.origin.copy_(sample.median(0).values)
        self.scale.copy_(sample.std(0).clamp_min(.1))
        normal=(sample-self.origin)/self.scale
        chosen=[int(normal.square().sum(-1).argmin())]
        distance=(normal-normal[chosen[0]]).square().sum(-1)
        for _ in range(self.capacity-1):
            index=int(distance.argmax());chosen.append(index)
            distance=torch.minimum(distance,(normal-normal[index]).square().sum(-1))
        self.anchors.copy_(sample[chosen])
        d=torch.cdist(normal[chosen],normal[chosen]);d.fill_diagonal_(float('inf'))
        self.radius.copy_(d.min(-1).values.median().clamp(.25,2.))
        self.initialized.fill_(True)

    def assignments(self,x):
        if not bool(self.initialized):raise ValueError('Object candidates not initialized')
        normal=(x-self.origin)/self.scale
        anchors=(self.anchors-self.origin)/self.scale
        distance=(normal[:,None]-anchors[None]).square().sum(-1)
        logits=self.identity(encode(normal))-distance/(2*self.radius.square())
        return logits.masked_fill(~self.active[None],-1e4).softmax(-1)

    def grouped(self,x,t,p):
        # A step queries one timestamp for all kernels, matching PIDG's renderer.
        time=t.reshape(-1)[:1,None]
        params=self.motion(encode(time)).reshape(self.capacity,7)
        q=params[:,:4]+params.new_tensor([1.,0,0,0])
        r=rotation(q)
        translation=2*params[:,4:].tanh()
        # Weighted transforms, avoiding a [N,K,3,3] intermediate.
        transform=torch.einsum('nk,kij->nij',p,r)
        shift=self.anchors+translation-torch.einsum('kij,kj->ki',r,self.anchors)
        return torch.einsum('nij,nj->ni',transform,x)+p@shift

    def forward(self,x,t,warp,blend):
        if blend==0:return warp,None
        if not 0<blend<=1:raise ValueError('Blend must lie in [0,1]')
        p=self.assignments(x)
        original=torch.einsum('nij,nj->ni',rotation(warp[:,:4]),x)+warp[:,4:]
        group=self.grouped(x,t,p)
        delta=blend*(group-original)
        result=torch.cat((warp[:,:4],warp[:,4:]+delta),-1)
        ids=torch.arange(min(1024,len(x)),device=x.device)*len(x)//min(1024,len(x))
        return result,{'p':p[ids],'residual':original[ids]-group[ids]}

    def regularizer(self,state,alignment=1e-3,sharpness=5e-5,complexity=1e-6):
        p=state['p'].clamp_min(1e-8)
        fit=F.smooth_l1_loss(state['residual'],torch.zeros_like(state['residual']),beta=.1)
        entropy=-(p*p.log()).sum(-1).mean()
        mass=p.mean(0);cost=(mass+1e-8).sqrt().sum()
        return alignment*fit+sharpness*entropy+complexity*cost,{
            'trajectory_residual_huber':fit.detach(),'assignment_entropy':entropy.detach(),
            'occupancy_sqrt_cost':cost.detach(),'max_assignment_mean':p.max(-1).values.mean().detach(),
            'effective_slots_1pct':(mass>.01).sum().detach(),'active_candidates':self.active.sum().detach()}

    @torch.no_grad()
    def prune_unused(self,x,min_mass=.001):
        p=self.assignments(x);mass=p.mean(0)
        keep=self.active&(mass>=min_mass)
        if not keep.any():keep[mass.argmax()]=True
        self.active.copy_(keep)
        return mass
