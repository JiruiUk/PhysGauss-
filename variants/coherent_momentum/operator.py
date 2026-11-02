"""Differentiable, normalized local polynomial momentum operator.

No material law or force is introduced. rho=1 and the original predicted stress
plus force_sigma*dev(sym(grad(v))) are retained. All derivatives are current-world
derivatives. The velocity Hessian replaces two successive first-order fits.
"""
import torch


def interval(t, h):
    if not 0 < h < .25 or not 0 <= t <= 1:
        raise ValueError('Invalid normalized interval')
    center = min(1-h, max(h, float(t)))
    return center-h, center, center+h


def kinematics(function, t, h):
    times = interval(t, h)
    # Cast before differencing, not after. Model evaluations remain FP32.
    xm, x, xp = [function(s).double() for s in times]
    vm, vp = (x-xm)/h, (xp-x)/h
    return x, .5*(vm+vp), (vp-vm)/h, xm, xp, times


def polynomial(x, values, anchor, neighbor, ridge=1e-5, rank_ratio=1e-6):
    """Anchored quadratic WLS, scalar distance normalization, FP64 solve.

    values: [nodes, channels]. Return Jacobian [anchors,channels,xyz] and
    Hessian [anchors,channels,xyz,xyz], plus detached rank diagnostics.
    """
    x, values = x.double(), values.double()
    dx = x[neighbor]-x[anchor, None]
    dy = values[neighbor]-values[anchor, None]
    scale = dx.detach().square().sum(-1).mean(-1).sqrt().clamp_min(1e-12)
    u = dx/scale[:, None, None]
    a,b,c = u.unbind(-1)
    phi = torch.stack([a,b,c,.5*a*a,.5*b*b,.5*c*c,a*b,a*c,b*c], -1)
    weight = 1/(u.detach().square().sum(-1)+.1)
    matrix = torch.einsum('nk,nki,nkj->nij', weight, phi, phi)
    eigen = torch.linalg.eigvalsh(matrix.detach())
    ratio = eigen[:, 0].clamp_min(0)/eigen[:, -1].clamp_min(1e-30)
    valid = ratio > rank_ratio
    eye = torch.eye(9, device=x.device, dtype=x.dtype)
    matrix = matrix+ridge*matrix.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None,None]*eye/9
    rhs = torch.einsum('nk,nki,nkc->nic', weight, phi, dy)
    beta = torch.linalg.solve(matrix, rhs).transpose(-1,-2)
    gradient = beta[:,:,:3]/scale[:,None,None]
    q = beta[:,:,3:]/scale[:,None,None].square()
    xx,yy,zz,xy,xz,yz = q.unbind(-1)
    hessian = torch.stack([xx,xy,xz,xy,yy,yz,xz,yz,zz],-1).reshape(*q.shape[:2],3,3)
    return gradient, hessian, valid, ratio


def divergence(x, velocity, stress6, coefficient, anchor, neighbor, ridge=1e-5):
    grad, hessian, valid, rank = polynomial(x,torch.cat([velocity,stress6],-1),anchor,neighbor,ridge)
    # Stress components xx,yy,zz,xy,xz,yz; sigma_ij,j.
    s = grad[:,3:]
    predicted = torch.stack([s[:,0,0]+s[:,3,1]+s[:,4,2],
        s[:,3,0]+s[:,1,1]+s[:,5,2],s[:,4,0]+s[:,5,1]+s[:,2,2]],-1)
    hv = hessian[:,:3]
    laplacian = hv.diagonal(dim1=-2,dim2=-1).sum(-1)
    # grad(div(v))_i = sum_j d_i d_j v_j.
    grad_div = torch.stack([sum(hv[:,j,i,j] for j in range(3)) for i in range(3)],-1)
    rate = coefficient*(.5*laplacian+grad_div/6)
    return predicted+rate, valid, {'predicted':predicted,'rate':rate,'rank_ratio':rank}


def momentum(x, velocity, stress, acceleration, coefficient, anchor, neighbor, ridge=1e-5):
    force, valid, parts = divergence(x,velocity,stress,coefficient,anchor,neighbor,ridge)
    return acceleration[anchor]-force, valid, parts
