import torch
from physics.contact import contact_force,plane_sdf
from physics.forces import gravity_force


def test_penetration_pushes_out_and_gravity_units():
    x=torch.tensor([[0.,-.1,0.],[0.,.1,0.]],requires_grad=True)
    f,loss=contact_force(x,plane_sdf(x,offset=0),100.)
    torch.testing.assert_close(f,torch.tensor([[0.,10.,0.],[0.,0.,0.]]))
    assert loss>0
    loss.backward();assert x.grad[0,1]<0
    torch.testing.assert_close(gravity_force(torch.full((2,1),2.)),torch.tensor([[0.,-19.62,0.]]).expand(2,3))
