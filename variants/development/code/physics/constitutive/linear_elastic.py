import torch
from torch import nn
from torch.nn import functional as Fnn


def lame_parameters(E, nu):
    if (E.detach() <= 0).any() or ((nu.detach() < 0) | (nu.detach() >= 0.5)).any():
        raise ValueError("Require E>0 and 0<=nu<0.5")
    return E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))


class LinearElasticMaterial(nn.Module):
    def __init__(self, youngs_modulus=1., poisson_ratio=.3):
        super().__init__()
        self.log_E = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(youngs_modulus)))))
        fraction = torch.tensor(poisson_ratio/.49)
        self.raw_nu = nn.Parameter(torch.logit(fraction))

    def parameters_physical(self):
        return Fnn.softplus(self.log_E)+1e-6, .49*torch.sigmoid(self.raw_nu)

    def forward(self, F, youngs_modulus=None, poisson_ratio=None):
        assert F.ndim == 3 and F.shape[-2:] == (3, 3)
        default_E, default_nu = self.parameters_physical()
        E = default_E if youngs_modulus is None else youngs_modulus
        nu = default_nu if poisson_ratio is None else poisson_ratio
        lam, mu = lame_parameters(E, nu)
        identity = torch.eye(3, device=F.device, dtype=F.dtype)
        strain = .5*(F+F.transpose(-1,-2))-identity
        trace = strain.diagonal(dim1=-2, dim2=-1).sum(-1)
        return lam.reshape(-1,1,1)*trace[:,None,None]*identity + 2*mu.reshape(-1,1,1)*strain
