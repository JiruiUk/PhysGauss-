"""Bounded three-knot inverse-depth proposal, independently validated in RGB."""
import torch
from ..adaptive_static.geometry import source_cost


def quadratic_offset(cost):
    """Costs[3,N] in grid order; retain nonconvex/invalid proposals at zero."""
    left,center,right = cost.unbind(0)
    curvature = left-2*center+right
    valid = torch.isfinite(cost).all(0)&(curvature > 0)
    safe = torch.where(valid,curvature,torch.ones_like(curvature))
    offset = .5*(left-right)/safe
    offset = torch.where(valid,offset.clamp(-.5,.5),torch.zeros_like(offset))
    return offset,valid


def accept_update(old,new,fit_old,fit_new,held_old,held_new,valid,enabled=False):
    if not enabled: return old,torch.zeros_like(valid)
    accepted = valid&torch.isfinite(new)&(new > 0)&torch.isfinite(fit_new)
    accepted &= (fit_new <= fit_old+1e-6)&torch.isfinite(held_new).all(0)
    accepted &= (held_new <= held_old+1e-6).all(0)
    accepted &= new != old
    return torch.where(accepted,new,old),accepted


@torch.no_grad()
def refine(uv,z,features,texture,views,k,index,sources,near,far,samples=192,enabled=False):
    if not enabled: return z,torch.zeros(len(z),device=z.device,dtype=torch.bool)
    assert len(sources) == 6 and samples == 192
    q = uv.long(); reference = features[index,:,q[:,1],q[:,0]]
    inverse = torch.linalg.inv(views[index]); rays = torch.cat([uv,torch.ones_like(uv[:,:1])],1)@torch.linalg.inv(k).T
    def points(depth):
        local = rays[None]*depth[...,None]
        return local@inverse[:3,:3].T+inverse[:3,3]
    def fit(depth):
        cost = torch.stack([source_cost(points(depth),reference,features[j],texture[j],views[j],k) for j in sources[:4]])
        return cost.sort(0).values[:3].mean(0)
    step = (1/far-1/near)/(samples-1); inv = 1/z
    knot_inv = inv[None]+torch.tensor([-1,0,1],device=z.device,dtype=z.dtype)[:,None]*step
    costs = fit(1/knot_inv)
    offset,valid = quadratic_offset(costs)
    cell = ((inv-1/near)/step).round()
    valid &= (cell > 0)&(cell < samples-1)
    proposed = 1/(inv+step*offset)
    fitted = fit(proposed[None])[0]
    def held(depth):
        return torch.stack([source_cost(points(depth[None]),reference,features[j],texture[j],views[j],k)[0] for j in sources[4:]])
    return accept_update(z,proposed,costs[1],fitted,held(z),held(proposed),valid,True)
