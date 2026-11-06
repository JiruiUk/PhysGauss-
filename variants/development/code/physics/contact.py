import torch
from .differential import derivative


def plane_sdf(x, normal=(0.,1.,0.), offset=-1.3):
    normal = x.new_tensor(normal)
    if normal.norm().item() == 0: raise ValueError("Zero plane normal")
    normal = normal/normal.norm()
    return (x*normal).sum(-1,keepdim=True)-offset, normal.expand_as(x)


def contact_force(x, sdf, stiffness=100.):
    if stiffness < 0: raise ValueError("Negative contact stiffness")
    if isinstance(sdf, tuple):
        distance, normal = sdf
    else:
        distance = sdf(x)
        normal = derivative(distance,x)
    assert distance.shape == (len(x),1) and normal.shape == x.shape
    penetration = distance.clamp_max(0)
    return -stiffness*penetration*normal, penetration.square().mean()
