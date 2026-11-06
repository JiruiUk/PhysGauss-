import torch
from physics.constitutive.linear_elastic import LinearElasticMaterial


def test_identity_and_simple_shear_stress():
    material=LinearElasticMaterial().double()
    I=torch.eye(3,dtype=torch.float64)[None]
    torch.testing.assert_close(material(I),I*0)
    F=I.clone();F[:,0,1]=.1
    E,nu=material.parameters_physical();mu=E/(2*(1+nu))
    stress=material(F)
    torch.testing.assert_close(stress[:,0,1],(.1*mu).expand(1))
    assert E>0 and 0<nu<.49
    stress.square().sum().backward()
    assert material.log_E.grad.abs()>0
