import torch


def gravity_force(rho, vector=(0.,-9.81,0.)):
    assert rho.ndim == 2 and rho.shape[1] == 1
    return rho*rho.new_tensor(vector).reshape(1,3)


def external_force(x, rho, gravity=None, contact=None, actuation=None, residual=None):
    force = torch.zeros_like(x)
    if gravity is not None: force = force + gravity_force(rho, gravity)
    for item in (contact, actuation, residual):
        if item is not None:
            assert item.shape == x.shape
            force = force+item
    return force
