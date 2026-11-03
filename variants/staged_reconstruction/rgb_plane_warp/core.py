"""Data-derived local slopes and differentiable raw-RGB support warping."""
import numpy as np
import torch
import torch.nn.functional as F
from ..adaptive_static.geometry import project_torch
from ..rgb_prior_proposals.core import accept_rgb


def fit_slopes(depth,uv,protocol):
    """No labels/GT. Fit/check disjoint original D1 neighbors, excluding center."""
    offsets=np.array([(dx,dy) for dy in range(-2,3) for dx in range(-2,3) if dx or dy])
    parity=(offsets.sum(1)%2)==0
    design=np.c_[offsets,np.ones(24)]
    n=len(uv); coefficients=np.full((n,3),np.nan,np.float64)
    fit_count=np.zeros(n,np.int32); held_count=fit_count.copy()
    residual=np.full(n,np.nan,np.float64); admitted=np.zeros(n,bool)
    h,w=depth.shape
    for i,(x,y) in enumerate(uv):
        if x<2 or y<2 or x>=w-2 or y>=h-2: continue
        z=depth[y+offsets[:,1],x+offsets[:,0]].astype(np.float64)
        positive=np.isfinite(z)&(z>0); fit=positive&parity; held=positive&~parity
        fit_count[i],held_count[i]=fit.sum(),held.sum()
        if fit_count[i]<protocol['slope_min_fit_neighbors'] or held_count[i]<protocol['slope_min_held_neighbors']: continue
        rank=protocol['slope_design_matrix_rank']
        if np.linalg.matrix_rank(design[fit])!=rank or np.linalg.matrix_rank(design[held])!=rank: continue
        coef=np.linalg.lstsq(design[fit],1/z[fit],rcond=None)[0]; coefficients[i]=coef
        predicted_inverse=design[positive]@coef
        if not np.isfinite(coef).all() or coef[2]<=0 or (predicted_inverse<=0).any(): continue
        errors=np.r_[np.abs(1/predicted_inverse-z[positive]),abs(1/coef[2]-float(depth[y,x]))]
        residual[i]=errors.max()
        admitted[i]=np.isfinite(residual[i]) and residual[i]<=protocol['slope_fit_held_and_center_max_abs_depth_residual']
    return {'coefficients':coefficients,'slope_admitted':admitted,'fit_neighbors':fit_count,
            'held_neighbors':held_count,'max_depth_residual':residual}


def plane_support(uv,depth,slopes,view,k,near,far):
    """Center depth[D,N] and inverse-depth slope[N,2] -> world[D,N,25,3]."""
    offsets=torch.tensor([(dx,dy) for dy in range(-2,3) for dx in range(-2,3)],dtype=uv.dtype,device=uv.device)
    inverse_depth=1/depth[...,None]+(slopes@offsets.T)[None]
    good=torch.isfinite(inverse_depth)&(inverse_depth>0)
    safe=torch.where(good,inverse_depth,torch.ones_like(inverse_depth))
    patch_depth=1/safe
    valid=(good&(patch_depth>=near)&(patch_depth<=far)).all(-1)
    xy=uv[:,None]+offsets[None]
    rays=torch.cat([xy,torch.ones_like(xy[...,:1])],-1)@torch.linalg.inv(k).T
    inverse=torch.linalg.inv(view)
    points=(rays[None]*patch_depth[...,None])@inverse[:3,:3].T+inverse[:3,3]
    return points,valid


def warped_cost(points,plane_valid,reference,reference_valid,source,view,k):
    """Sample RGB first, then normalize the aligned support; no feature interpolation."""
    d,n,p,_=points.shape; assert p==25
    grid,good,_,_=project_torch(points,view,k,*source.shape[-2:])
    sampled=F.grid_sample(source[None],grid.reshape(1,d,n*25,2),align_corners=True,
                          padding_mode='zeros')[0].reshape(3,d,n,25)
    gray=.299*sampled[0]+.587*sampled[1]+.114*sampled[2]
    centered=gray-gray.mean(-1,keepdim=True)
    norm=centered.square().sum(-1,keepdim=True).sqrt()
    patch=centered/norm.clamp_min(1e-5)
    center=sampled[:,:,:,12]
    chroma=center/center.square().sum(0,keepdim=True).sqrt().clamp_min(1e-5)
    compatible=(chroma*reference[-3:,None]).sum(0)>=.55
    valid=plane_valid&reference_valid[None]&good.all(-1)&(norm[...,0]/5>=.008)&(center.mean(0)>.025)&compatible
    cost=((1-(patch*reference[:-3].T[None]).sum(-1).clamp(-1,1))/2).clamp(0,1)
    return torch.where(valid,cost,torch.ones_like(cost))


@torch.no_grad()
def propose(uv,old,slopes=None,admitted=None,images=None,features=None,texture=None,
            views=None,k=None,index=None,sources=None,near=None,far=None,enabled=False):
    if not enabled: return old,{'accepted':torch.zeros_like(old,dtype=torch.bool)}
    assert len(sources)==6 and torch.isfinite(old).all() and (old>0).all()
    empty=torch.full_like(old,float('nan')); mask=torch.zeros_like(old,dtype=torch.bool)
    diag={'accepted':mask.clone(),'eligible':mask.clone(),'proposed':empty.clone(),
          'fit_old':empty.clone(),'fit_new':empty.clone(),'held_old':empty.repeat(2,1),'held_new':empty.repeat(2,1)}
    ids=torch.nonzero(admitted).flatten()
    if not len(ids): return old.clone(),diag
    xy,z,s=uv[ids],old[ids],slopes[ids]
    points_old,valid_old=plane_support(xy,z[None],s,views[index],k,near,far)
    ids=ids[valid_old[0]]
    if not len(ids): return old.clone(),diag
    xy,z,s=uv[ids],old[ids],slopes[ids]
    diag['eligible'][ids]=True
    ref=features[index,:,xy[:,1].long(),xy[:,0].long()]
    refvalid=texture[index,xy[:,1].long(),xy[:,0].long()]
    def costs(depth,chosen):
        points,valid=plane_support(xy,depth,s,views[index],k,near,far)
        cs=torch.stack([warped_cost(points,valid,ref,refvalid,images[j],views[j],k) for j in chosen])
        return cs,valid
    grid=1/torch.linspace(1/near,1/far,192,dtype=z.dtype,device=z.device)
    cs,valid=costs(grid[:,None].expand(-1,len(ids)),sources[:4])
    fit=cs.sort(0).values[:3].mean(0).masked_fill(~valid,float('inf'))
    best,which=fit.min(0); proposed=grid[which]
    old_fit=costs(z[None],sources[:4])[0].sort(0).values[:3].mean(0)[0]
    both=torch.stack([z,proposed]); held,_=costs(both,sources[4:])
    held_old,held_new=held[:,0],held[:,1]
    updated,accepted=accept_rgb(z,proposed,old_fit,best,held_old,held_new,valid.any(0)&torch.isfinite(best),True)
    result=old.clone(); result[ids]=updated
    diag['accepted'][ids]=accepted
    diag['proposed'][ids]=torch.where(valid.any(0),proposed,torch.full_like(proposed,float('nan')))
    diag['fit_old'][ids],diag['fit_new'][ids]=old_fit,best
    diag['held_old'][:,ids],diag['held_new'][:,ids]=held_old,held_new
    return result,diag
