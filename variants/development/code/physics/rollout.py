import torch
from .contact import plane_sdf, contact_force


def symplectic_euler(x,v,mass,force,dt):
    assert x.shape == v.shape == force.shape and mass.shape == (len(x),1)
    if dt <= 0 or (mass.detach() <= 0).any(): raise ValueError("Positive dt and mass required")
    updated_velocity = v+dt*force/mass
    return x+dt*updated_velocity,updated_velocity


def rollout(x,v,mass,force_model,dt,steps):
    if steps < 1: raise ValueError("At least one rollout step required")
    positions,velocities = [x],[v]
    for step in range(steps):
        x,v = symplectic_euler(x,v,mass,force_model(x,v,step*dt),dt)
        positions.append(x)
        velocities.append(v)
    return torch.stack(positions),torch.stack(velocities)


def particle_deformation_gradient(X,x,k=16):
    from .weak_form import build_neighbor_graph
    graph = build_neighbor_graph(X,k)
    dX = X[graph]-X[:,None]
    dx = x[graph]-x[:,None]
    moment = torch.einsum("nki,nkj->nij",dX,dX)
    cross = torch.einsum("nki,nkj->nij",dx,dX)
    regularizer = 1e-6*moment.diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(1e-8)
    identity = torch.eye(3,device=x.device,dtype=x.dtype)
    # Identity-preserving Tikhonov stabilization.
    return (cross+regularizer[:,None,None]*identity)@torch.linalg.inv(moment+regularizer[:,None,None]*identity)


def closed_particle_force(X,x,volumes0,material,gravity=None,contact=None,rho0=1.):
    F = particle_deformation_gradient(X,x)
    J = torch.linalg.det(F)[:,None]
    if (J.detach() <= 0).any(): raise ValueError("Particle rollout inverted a reference cell")
    stress = material(F)
    volumes = volumes0*J
    delta = x[:,None]-x[None]
    with torch.no_grad():
        h = torch.cdist(X,X).topk(min(16,len(X)),largest=False).values[:,-1].median().clamp_min(1e-3)
    W = torch.exp(-delta.square().sum(-1)/(2*h.square()))/((2*torch.pi)**1.5*h.pow(3))
    grad = -W[...,None]*delta/h.square()
    # Pairwise anti-symmetric force conserves total linear momentum in free space.
    pair_stress = stress[:,None]+stress[None]
    force = torch.einsum("ijab,ijb->ija",pair_stress,grad)
    force = (force*volumes[:,None]*volumes[None]).sum(1)
    if gravity is not None: force = force+rho0*volumes0*x.new_tensor(gravity)
    if contact is not None:
        force = force+contact_force(x,plane_sdf(x,**contact["plane"]),contact["stiffness"])[0]*volumes
    return force
