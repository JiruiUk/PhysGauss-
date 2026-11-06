"""Cubic B-spline in se(3), exponentiated to a continuous SE(3) curve.

This is a Lie-algebra spline, not a claim of a geodesic/cumulative group B-spline.
Canonical-position-dependent controls are constant in time and anchored at t=0.
"""
import torch
from torch import nn


def bspline_weights(t, controls=8):
    if controls < 4: raise ValueError("Cubic spline needs at least four controls")
    t = t.reshape(-1)
    if ((t.detach()<0) | (t.detach()>1)).any():
        raise ValueError("Spline queries must stay in [0,1]; use physical rollout for extrapolation")
    interior = torch.linspace(0,1,controls-2,device=t.device,dtype=t.dtype)[1:-1]
    knots = torch.cat([t.new_zeros(4),interior,t.new_ones(4)])
    # Right endpoint uses its left limit; t derivative remains the one-sided derivative.
    q = torch.where(t==1, t-t.new_tensor(torch.finfo(t.dtype).eps), t)
    basis = ((q[:,None]>=knots[:-1]) & (q[:,None]<knots[1:])).to(t.dtype)
    for degree in range(1,4):
        count = len(knots)-degree-1
        left_den = knots[degree:degree+count]-knots[:count]
        right_den = knots[degree+1:degree+1+count]-knots[1:1+count]
        left = torch.where(left_den>0, (q[:,None]-knots[:count])/left_den.clamp_min(1e-12),0)
        right = torch.where(right_den>0, (knots[degree+1:degree+1+count]-q[:,None])/right_den.clamp_min(1e-12),0)
        basis = left*basis[:,:count]+right*basis[:,1:count+1]
    return basis


def se3_exp(twist):
    assert twist.ndim == 2 and twist.shape[-1] == 6
    omega,u = twist[:,:3],twist[:,3:]
    theta2 = omega.square().sum(-1,keepdim=True)
    theta = theta2.clamp_min(1e-12).sqrt()
    small = theta2 < 1e-6
    qw = torch.where(small,1-theta2/8+theta2.square()/384,torch.cos(theta/2))
    qfactor = torch.where(small,.5-theta2/48+theta2.square()/3840,torch.sin(theta/2)/theta)
    B = torch.where(small,.5-theta2/24+theta2.square()/720,(1-torch.cos(theta))/theta.square())
    C = torch.where(small,1/6-theta2/120+theta2.square()/5040,(theta-torch.sin(theta))/theta.pow(3))
    cross = torch.linalg.cross(omega,u)
    translation = u+B*cross+C*torch.linalg.cross(omega,cross)
    return torch.cat([qw,qfactor*omega],-1),translation


class ContinuousSE3Motion(nn.Module):
    def __init__(self, controls=8):
        super().__init__()
        self.controls = controls
        self.encoder = nn.Sequential(nn.Linear(3,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.field = nn.Linear(64,controls*6)
        nn.init.zeros_(self.field.weight)
        nn.init.zeros_(self.field.bias)

    def forward(self,X,t):
        controls = self.field(self.encoder(X)).reshape(-1,self.controls,6)
        basis = bspline_weights(t,self.controls)-bspline_weights(torch.zeros_like(t),self.controls)
        twist = torch.einsum("nk,nkc->nc",basis,controls)
        return se3_exp(twist)

    def apply_motion(self, legacy_motion, X, t):
        q,b = self(X,t)
        return torch.cat([q,b],-1)
