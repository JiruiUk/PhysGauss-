import warnings
import torch
from .linear_elastic import LinearElasticMaterial, lame_parameters


class NeoHookeanMaterial(LinearElasticMaterial):
    def energy(self, F, youngs_modulus=None, poisson_ratio=None):
        E, nu = self.parameters_physical()
        E = E if youngs_modulus is None else youngs_modulus
        nu = nu if poisson_ratio is None else poisson_ratio
        lam, mu = lame_parameters(E, nu)
        J = torch.linalg.det(F)
        if (J.detach() <= 0).any():
            warnings.warn("Neo-Hookean stabilized log evaluated at invalid J", RuntimeWarning)
        logJ = J.clamp_min(1e-6).log()
        return mu.reshape(-1)*.5*(F.square().sum((-2,-1))-3) - mu.reshape(-1)*logJ + .5*lam.reshape(-1)*logJ.square()

    def forward(self, F, youngs_modulus=None, poisson_ratio=None):
        assert F.ndim == 3 and F.shape[-2:] == (3,3)
        E, nu = self.parameters_physical()
        E = E if youngs_modulus is None else youngs_modulus
        nu = nu if poisson_ratio is None else poisson_ratio
        lam, mu = lame_parameters(E, nu)
        J = torch.linalg.det(F)[:,None,None]
        if (J.detach() <= 0).any():
            warnings.warn("Neo-Hookean stabilized stress evaluated at invalid J", RuntimeWarning)
        safe = J.clamp_min(1e-6)
        I = torch.eye(3, device=F.device, dtype=F.dtype)
        return (mu.reshape(-1,1,1)*(F@F.transpose(-1,-2)-I) + lam.reshape(-1,1,1)*safe.log()*I)/safe
