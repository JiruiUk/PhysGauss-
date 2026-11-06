import torch
from physics.rollout import rollout,closed_particle_force,particle_deformation_gradient
from physics.constitutive import LinearElasticMaterial


def test_symplectic_translation_and_bounded_oscillator_energy():
    x=torch.tensor([[1.,0,0]],dtype=torch.float64)
    v=torch.tensor([[.1,.2,.3]],dtype=x.dtype)
    mass=torch.ones(1,1,dtype=x.dtype)
    positions,velocities=rollout(x,v,mass,lambda x,v,t:x*0,.01,100)
    torch.testing.assert_close(positions[-1],x+v)
    torch.testing.assert_close(velocities[-1],v)
    positions,velocities=rollout(x,v*0,mass,lambda x,v,t:-x,.01,1000)
    energy=.5*(positions.square()+velocities.square()).sum(-1)
    assert (energy.max()-energy.min())<.011


def test_reference_fit_and_internal_force_momentum_conservation():
    torch.manual_seed(4)
    X=torch.randn(24,3,dtype=torch.float64)
    F=particle_deformation_gradient(X,X)
    torch.testing.assert_close(F,torch.eye(3,dtype=X.dtype).expand(24,3,3),atol=1e-9,rtol=1e-9)
    x=(X*1.01).requires_grad_()
    material=LinearElasticMaterial().double()
    force=closed_particle_force(X,x,torch.ones(24,1,dtype=X.dtype)/24,material)
    torch.testing.assert_close(force.sum(0),torch.zeros(3,dtype=X.dtype),atol=1e-9,rtol=1e-9)
    force.square().mean().backward();assert torch.isfinite(x.grad).all()
