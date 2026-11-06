import pytest
import torch
from physics.conservation import density_from_jacobian, mass_conservation_loss, volume_preservation_loss


def test_exact_lagrangian_mass_and_positive_jacobians():
    J=torch.tensor([[.5],[1.],[2.]],requires_grad=True)
    rho=density_from_jacobian(3.,J)
    torch.testing.assert_close(rho*J,torch.full_like(J,3))
    assert mass_conservation_loss(rho,J,3.).item()<1e-12
    assert volume_preservation_loss(J)>0
    with pytest.raises(ValueError): density_from_jacobian(1.,-J)
