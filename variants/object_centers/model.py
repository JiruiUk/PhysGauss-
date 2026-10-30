"""Group centers and soft effective-volume overlap; no hard collision guarantee."""
import torch
from torch import nn


def weighted_centers(x, labels, weights, groups):
    if x.ndim!=2 or x.shape[-1]!=3 or labels.shape!=(len(x),) or weights.shape!=(len(x),):
        raise ValueError('Expected aligned[N,3], IDs[N], weights[N]')
    if groups<1 or labels.dtype!=torch.long or (labels < -1).any() or (labels>=groups).any():
        raise ValueError('Group IDs must be int64 in [-1, groups)')
    if not torch.isfinite(x).all() or not torch.isfinite(weights).all() or (weights<0).any():
        raise ValueError('Finite positions and nonnegative weights required')
    valid=(labels>=0)&(labels<groups)&(weights>0)
    mass=x.new_zeros(groups).index_add(0,labels[valid],weights[valid].to(x))
    if (mass<=0).any():raise ValueError('An object-center group is empty')
    first=x.new_zeros(groups,3).index_add(0,labels[valid],x[valid]*weights[valid,None].to(x))
    return first/mass[:,None]


def center_residual(x, labels, weights, groups):
    centers=weighted_centers(x,labels,weights,groups)
    selected=labels>=0
    residual=x.clone();residual[selected]=x[selected]-centers[labels[selected]]
    return centers,residual


class CenterSpline(nn.Module):
    """Piecewise-linear center trajectory with learned knot offsets.

    Initializes from explicitly supplied training-time center candidates. Endpoint
    clamping is not physical extrapolation. No Gaussian mass is assumed.
    """
    def __init__(self,times,centers):
        super().__init__()
        if times.ndim!=1 or len(times)<2 or centers.ndim!=3 or centers.shape[0]!=len(times) or centers.shape[-1]!=3 or \
           not torch.isfinite(centers).all() or not (times.diff()>0).all():
            raise ValueError('Finite centers and strictly increasing time knots required')
        self.register_buffer('times',times.clone())
        self.register_buffer('initial_centers',centers.clone())
        self.offsets=nn.Parameter(torch.zeros_like(centers))

    def forward(self,t):
        values=self.initial_centers+self.offsets
        t=t.to(self.times).flatten().clamp(self.times[0],self.times[-1])
        high=torch.searchsorted(self.times,t).clamp(1,len(self.times)-1);low=high-1
        ratio=(t-self.times[low])/(self.times[high]-self.times[low])
        return values[low]*(1-ratio[:,None,None])+values[high]*ratio[:,None,None]

    def smoothness(self):
        v=(self.initial_centers+self.offsets).diff(dim=0)/self.times.diff()[:,None,None]
        return v.diff(dim=0).square().mean() if len(v)>1 else v.sum()*0


def inherit_groups(labels,parent_indices,keep):
    if parent_indices.numel() and (parent_indices.min()<0 or parent_indices.max()>=len(labels)):
        raise ValueError('Invalid parent kernel index')
    expanded=torch.cat((labels,labels[parent_indices]))
    if keep.shape!=expanded.shape:raise ValueError('Topology mask is misaligned')
    return expanded[keep]


def cross_group_sphere_proxy(x,radii,labels,pairs,allowance=0.):
    """Soft overlap of detached effective radii at supplied candidate pairs.

    Infinite Gaussian tails are not solids. This is only an effective-volume
    diagnostic/penalty; sampled pairs and spheres cannot certify nonpenetration.
    """
    i,j=pairs.unbind(-1)
    valid=(labels[i]>=0)&(labels[j]>=0)&(labels[i]!=labels[j])
    if not valid.any():return x.sum()*0
    i,j=i[valid],j[valid];delta=x[i]-x[j]
    # Deterministic tie direction avoids zero gradients at coincident centers.
    axis=(i+j)%3;direction=torch.nn.functional.one_hot(axis,3).to(x)
    delta=delta+(delta.detach().square().sum(-1)<1e-12)[:,None]*direction*1e-6
    distance=delta.norm(dim=-1)
    radius=(radii[i]+radii[j]).detach()
    return (radius-distance-allowance).clamp_min(0).square().mean()
