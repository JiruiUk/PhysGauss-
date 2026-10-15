"""Objective compressible Neo-Hookean Cauchy stress and positive shared parameters."""
import torch
from torch import nn
from torch.nn import functional as functional


def vec6_to_symmat(value):
    """Component order is xx, yy, zz, xy, xz, yz (no engineering-shear factors)."""
    if value.shape[-1]!=6:raise ValueError('Expected six stress components')
    a,b,c,d,e,f=value.unbind(-1)
    return torch.stack((a,d,e,d,b,f,e,f,c),-1).reshape(*value.shape[:-1],3,3)


def symmat_to_vec6(value):
    if value.shape[-2:]!=(3,3):raise ValueError('Expected 3x3 stress matrices')
    if not torch.allclose(value,value.transpose(-1,-2)):
        raise ValueError('Stress matrix is not symmetric')
    return torch.stack((value[...,0,0],value[...,1,1],value[...,2,2],
                        value[...,0,1],value[...,0,2],value[...,1,2]),-1)


class MaterialParameters(nn.Module):
    def __init__(self,mu=1.,kappa=10.,learn=False):
        super().__init__()
        if mu<=1e-6 or kappa<=1e-6:raise ValueError('Positive material initializations required')
        def raw(value):
            value=torch.as_tensor(value,dtype=torch.float64)-1e-6
            return value+torch.log(-torch.expm1(-value))
        self.raw_mu=nn.Parameter(raw(mu),requires_grad=learn)
        self.raw_kappa=nn.Parameter(raw(kappa),requires_grad=learn)

    def forward(self):
        return {'mu':functional.softplus(self.raw_mu)+1e-6,
                'kappa':functional.softplus(self.raw_kappa)+1e-6}


class ConstitutiveModel(nn.Module):
    def forward(self,F,material_params):
        raise NotImplementedError


class NeoHookean(ConstitutiveModel):
    """W = mu/2 (J^(-2/3) tr(B)-3) + kappa/2 log(J)^2.

    kappa is the bulk modulus. Inputs with nonpositive J fail explicitly; callers
    must select valid F before evaluating. No absolute determinant is used.
    """
    def forward(self,F,material_params):
        if F.shape[-2:]!=(3,3):raise ValueError('Expected F[...,3,3]')
        J=torch.linalg.det(F)
        if not torch.isfinite(F).all() or (J<=0).any():
            raise ValueError('Neo-Hookean stress requires finite F and positive determinant')
        mu=torch.as_tensor(material_params['mu'],device=F.device,dtype=F.dtype)
        kappa=torch.as_tensor(material_params['kappa'],device=F.device,dtype=F.dtype)
        eye=torch.eye(3,device=F.device,dtype=F.dtype)
        B=F@F.transpose(-1,-2)
        dev=B-B.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]/3*eye
        return (mu*J.pow(-5/3))[...,None,None]*dev+(kappa*J.log()/J)[...,None,None]*eye
