import torch
from physics.constitutive import LinearElasticMaterial,NeoHookeanMaterial
from physics.constitutive.moe import ConstitutiveMoE
from physics.neural_residual import BoundedStressResidual
from models.material_field import MaterialField, material_smoothness


def test_closed_stress_material_field_and_mixture():
    X=torch.randn(8,3,dtype=torch.float64)
    F=torch.eye(3,dtype=X.dtype)[None]+.01*torch.randn(8,3,3,dtype=X.dtype)
    field=MaterialField().double()
    params=field(X)
    assert (params['E']>0).all() and ((params['nu']>0)&(params['nu']<.49)).all()
    for material in (LinearElasticMaterial().double(),NeoHookeanMaterial().double()):
        stress=material(F,params['E'],params['nu'])
        torch.testing.assert_close(stress,stress.transpose(-1,-2))
    model=ConstitutiveMoE().double()
    stress,weights,experts=model(F,F*0,X,params['E'],params['nu'])
    torch.testing.assert_close(weights.sum(-1),torch.ones(8,dtype=X.dtype))
    torch.testing.assert_close(stress,stress.transpose(-1,-2))
    residual=BoundedStressResidual().double()
    residual.net[-1].bias.data.fill_(10.)
    result,penalty,ratio=residual(stress,F,X)
    assert ratio.max()<=.20000001 and penalty>0
    assert material_smoothness(X,params['features'])>=0
    result.square().mean().backward()
    assert model.gate.weight.grad is not None
