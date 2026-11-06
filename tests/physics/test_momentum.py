import torch
from physics.momentum import strong_form_momentum_residual


def test_spatial_chain_rule_from_reference_gradient():
    X=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    F=torch.eye(3,dtype=X.dtype)[None].repeat(4,1,1)*2
    spatial_x=2*X
    # sigma_xx = x, sigma_yy=y, sigma_zz=z -> spatial divergence [1,1,1].
    stress=torch.diag_embed(spatial_x)
    rho=torch.ones(4,1,dtype=X.dtype)
    acceleration=torch.ones_like(X)
    residual=strong_form_momentum_residual(rho,acceleration,stress,X,F=F)
    torch.testing.assert_close(residual,torch.zeros_like(X))
