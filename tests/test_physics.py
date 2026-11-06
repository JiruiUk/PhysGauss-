import torch
from torch import nn
from physgauss.physics import momentum_residual, trajectory_residual


class AnalyticMaterial(nn.Module):
    def forward(self, x, t):
        # v=(x^2,0,0), sigma_xx=x^3: Dv/Dt-div(sigma)=2*x^3-3*x^2.
        v = torch.stack([x[:, 0].square(), x[:, 1]*0, x[:, 2]*0], -1)
        sigma = x.new_zeros(len(x), 3, 3)
        sigma[:, 0, 0] = x[:, 0]**3
        return v, sigma


def test_spatial_material_derivative_and_stress_divergence():
    x = torch.tensor([[0.2, 0.3, 0.1], [0.7, 0.4, -0.2]])
    result = momentum_residual(AnalyticMaterial(), x, 0.4)
    expected = torch.zeros_like(x)
    expected[:, 0] = 2*x[:, 0]**3 - 3*x[:, 0].square()
    torch.testing.assert_close(result, expected)


class TemporalMaterial(nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, t):
        v = self.rate*t.expand(-1, 3)
        return v, x[:, None, :].expand(-1, 3, -1)*0


def test_temporal_derivative_density_body_force_and_second_order_gradients():
    material = TemporalMaterial()
    residual = momentum_residual(material, torch.ones(3, 3), 0.4, density=3,
                                 body_acceleration=[1, 1, 1])
    torch.testing.assert_close(residual, torch.full((3, 3), 3.0))
    residual.square().mean().backward()
    torch.testing.assert_close(material.rate.grad, torch.tensor(18.0))


class Translation(nn.Module):
    def displacement(self, x, t):
        return t.expand(-1, 3)*x.new_tensor([1, 2, 3])


class ConstantVelocity(nn.Module):
    def forward(self, x, t):
        return x.new_tensor([1, 2, 3]).expand_as(x), x.new_zeros(len(x), 3, 3)


def test_trajectory_velocity_matches_translation():
    residual = trajectory_residual(Translation(), ConstantVelocity(), torch.rand(4, 3), 0.7)
    torch.testing.assert_close(residual, torch.zeros_like(residual))
