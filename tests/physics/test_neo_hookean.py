import torch
from physics.constitutive.neo_hookean import NeoHookeanMaterial


def test_energy_stress_consistency_and_rotation_objectivity():
    material=NeoHookeanMaterial().double()
    I=torch.eye(3,dtype=torch.float64)[None]
    torch.testing.assert_close(material(I),I*0)
    F=(I+torch.tensor([[[.1,.02,0],[0,-.03,0],[0,0,.04]]],dtype=I.dtype)).requires_grad_()
    P=torch.autograd.grad(material.energy(F).sum(),F,create_graph=True)[0]
    expected=P@F.transpose(-1,-2)/torch.linalg.det(F)[:,None,None]
    torch.testing.assert_close(material(F),expected)
    rotation=I.clone();rotation[0,:2,:2]=torch.tensor([[0.,-1],[1,0]],dtype=I.dtype)
    torch.testing.assert_close(material(rotation),I*0)
