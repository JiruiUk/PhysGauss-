import torch
from variants.coverage_tuning.model import local_affine_loss, neighborhoods


def cloud():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(50,3,generator=g,dtype=torch.float64)
    return x, neighborhoods(x,0,count=10,radius=100)


def test_full_affine_stretch_shear_rotation_translation_is_not_penalized():
    x,(a,n,v) = cloud()
    A = torch.tensor([[2.,.3,0.],[0.,.7,.2],[.2,0.,1.]],dtype=x.dtype)
    y = x@A+3
    assert local_affine_loss(x,y,a,n,v) < 1e-25


def test_non_affine_motion_gets_finite_gradients_but_no_basis_gradient():
    x,(a,n,v) = cloud(); x.requires_grad_()
    y = (x.detach()+.2*x.detach().square()).requires_grad_()
    loss = local_affine_loss(x,y,a,n,v); loss.backward()
    assert loss > 0 and torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0
    assert x.grad is None


def test_neighbor_gating_does_not_connect_touching_objects():
    x = torch.randn(40,3)*.01; labels = torch.arange(40)//20
    a,n,v = neighborhoods(x,0,count=40,radius=1,object_ids=labels)
    assert torch.all((labels[a,None]==labels[n])[v])


def test_planar_or_empty_neighborhoods_are_finite():
    x,(a,n,v) = cloud(); x[:,2]=0
    y = x.clone().requires_grad_()
    loss = local_affine_loss(x,y,a,n,v); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(y.grad).all()
