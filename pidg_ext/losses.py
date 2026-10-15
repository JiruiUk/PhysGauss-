"""Constitutive consistency with the existing PIDG Cauchy-stress prediction."""
import torch
from torch.nn import functional as functional
from .constitutive import NeoHookean,vec6_to_symmat


def predicted_stress(coords,indices,velocity_net,force_sigma):
    velocity,components=velocity_net(coords,indices)
    grad_v=torch.stack([torch.autograd.grad(velocity[:,i].sum(),coords,
                        create_graph=True,retain_graph=True)[0][:,:3] for i in range(3)],dim=1)
    strain=.5*(grad_v+grad_v.transpose(-1,-2))
    eye=torch.eye(3,device=coords.device,dtype=coords.dtype)
    dev=strain-strain.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None,None]/3*eye
    return vec6_to_symmat(components)+force_sigma*dev


def constitutive_consistency(sigma_pred,F,params,confidence,valid):
    sigma_const=torch.zeros_like(sigma_pred)
    if valid.any():
        selected={key:torch.as_tensor(value,device=F.device,dtype=F.dtype) for key,value in params.items()}
        selected={key:value[valid] if value.ndim else value for key,value in selected.items()}
        sigma_const[valid]=NeoHookean()(F[valid],selected)
    error=functional.huber_loss(sigma_pred,sigma_const,reduction='none').mean(dim=(-1,-2))
    raw=(error*confidence.detach()*valid).mean()
    if not torch.isfinite(raw):raise FloatingPointError('Nonfinite constitutive loss')
    return raw,sigma_const
