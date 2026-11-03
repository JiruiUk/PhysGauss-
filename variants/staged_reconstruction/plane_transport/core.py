"""Transport only admitted source planes; all identities remain unknown."""
import torch
from ..adaptive_static.geometry import project_torch
from ..rgb_plane_warp.core import plane_support,warped_cost
from ..rgb_prior_proposals.core import accept_rgb


def world_point(uv,depth,view,k):
    inverse=torch.linalg.inv(view)
    rays=torch.cat([uv,torch.ones_like(uv[:,:1])],1)@torch.linalg.inv(k).T
    return (rays*depth[:,None])@inverse[:3,:3].T+inverse[:3,3]


def source_plane(uv,depth,slopes,view,k):
    covector=torch.cat([slopes,(1/depth-(slopes*uv).sum(1))[:,None]],1)
    normal=(covector@k)@view[:3,:3]
    return world_point(uv,depth,view,k),normal


def ray_plane_hit(origin,rays,anchor,normal,epsilon=1e-8):
    norm=normal.square().sum(1).sqrt()
    n=normal/norm.clamp_min(epsilon)[:,None]
    denominator=(n*rays).sum(1)
    valid=torch.isfinite(normal).all(1)&(norm>epsilon)&torch.isfinite(denominator)&(denominator.abs()>epsilon)
    safe=torch.where(valid,denominator,torch.ones_like(denominator))
    z=((anchor-origin)*n).sum(1)/safe
    return z,valid&torch.isfinite(z)&(z>0),n


def transport(uv,anchor_uv,anchor_depth,anchor_slopes,reference_view,source_view,k,height,width,near,far):
    anchor,normal=source_plane(anchor_uv,anchor_depth,anchor_slopes,source_view,k)
    inverse=torch.linalg.inv(reference_view); origin=inverse[:3,3]
    rays=(torch.cat([uv,torch.ones_like(uv[:,:1])],1)@torch.linalg.inv(k).T)@inverse[:3,:3].T
    z,valid,n=ray_plane_hit(origin,rays,anchor,normal)
    nref=n@reference_view[:3,:3].T
    local_anchor=anchor@reference_view[:3,:3].T+reference_view[:3,3]
    constant=(nref*local_anchor).sum(1)
    good_constant=torch.isfinite(constant)&(constant.abs()>1e-8)
    covector=(nref@torch.linalg.inv(k))/torch.where(good_constant,constant,torch.ones_like(constant))[:,None]
    slopes=covector[:,:2]
    hit=origin+rays*z[:,None]
    _,margin,projected,_=project_torch(hit[None],source_view,k,height,width)
    footprint=((projected[0]-anchor_uv).abs()<=2).all(1)
    _,plane_valid=plane_support(uv,z[None],slopes,reference_view,k,near,far)
    valid &= good_constant&torch.isfinite(slopes).all(1)&margin[0]&footprint&plane_valid[0]
    return z,slopes,valid,projected[0]


def incumbent_plane(uv,old,old_slopes,view,k,near,far):
    finite=torch.isfinite(old_slopes).all(1)
    cleaned=torch.nan_to_num(old_slopes)
    _,valid_initial=plane_support(uv,old[None],cleaned,view,k,near,far)
    fallback=~finite|~valid_initial[0]
    incumbent=torch.where(fallback[:,None],torch.zeros_like(cleaned),cleaned)
    points_old,valid_old=plane_support(uv,old[None],incumbent,view,k,near,far)
    return incumbent,fallback,points_old,valid_old


def query_hypotheses(uv,old,anchor_maps,views,k,index,sources,near,far,height,width):
    current_world=world_point(uv,old,views[index],k)
    depths=[]; slopes=[]; validity=[]; queries=[]; query_valid=[]; source_depth=[]; source_slopes=[]; projected_hits=[]
    for j in sources[:4]:
        _,margin,projected,_=project_torch(current_world[None],views[j],k,height,width)
        query=torch.round(torch.nan_to_num(projected[0])).long()
        x,y=query[:,0].clamp(0,width-1),query[:,1].clamp(0,height-1)
        z=anchor_maps['original'][j,y,x]; s=anchor_maps['slopes'][j,y,x]
        available=margin[0]&anchor_maps['admitted'][j,y,x]&torch.isfinite(z)&(z>0)&torch.isfinite(s).all(1)
        safe_z=torch.where(available,z,torch.ones_like(z)); safe_s=torch.where(available[:,None],s,torch.zeros_like(s))
        candidate,cs,good,hit=transport(uv,query.to(uv.dtype),safe_z,safe_s,views[index],views[j],k,height,width,near,far)
        depths.append(candidate); slopes.append(cs); validity.append(available&good)
        queries.append(query); query_valid.append(available); source_depth.append(z); source_slopes.append(s); projected_hits.append(hit)
    return {'source_uv':torch.stack(queries),'source_query_valid':torch.stack(query_valid),
        'source_original_depth':torch.stack(source_depth),'source_slopes':torch.stack(source_slopes),
        'raw_candidate_depth':torch.stack(depths),'raw_candidate_slopes':torch.stack(slopes),
        'candidate_valid':torch.stack(validity),'source_projected_hit':torch.stack(projected_hits)}


@torch.no_grad()
def propose(uv,old,old_slopes=None,anchor_maps=None,images=None,features=None,texture=None,
            views=None,k=None,index=None,sources=None,near=None,far=None,enabled=False):
    if not enabled: return old,{'accepted':torch.zeros_like(old,dtype=torch.bool)}
    assert len(sources)==6 and torch.isfinite(old).all() and (old>0).all()
    n=len(old); device=old.device; height,width=images.shape[-2:]
    incumbent,fallback,points_old,valid_old=incumbent_plane(uv,old,old_slopes,views[index],k,near,far)
    reference=features[index,:,uv[:,1].long(),uv[:,0].long()]
    ref_valid=texture[index,uv[:,1].long(),uv[:,0].long()]
    hyp=query_hypotheses(uv,old,anchor_maps,views,k,index,sources,near,far,height,width)
    raw_depth,raw_slopes,valid=hyp['raw_candidate_depth'],hyp['raw_candidate_slopes'],hyp['candidate_valid']
    # Invalid hypotheses retain their diagnostics but never enter a cost minimum.
    points=[]; plane_flags=[]
    for h in range(4):
        safe_z=torch.where(valid[h],raw_depth[h],old)
        safe_s=torch.where(valid[h,:,None],raw_slopes[h],incumbent)
        p,v=plane_support(uv,safe_z[None],safe_s,views[index],k,near,far)
        points.append(p[0]); plane_flags.append(v[0]&valid[h])
    points=torch.stack(points); plane_flags=torch.stack(plane_flags)
    def costs(p,v,chosen):
        return torch.stack([warped_cost(p,v,reference,ref_valid,images[j],views[j],k) for j in chosen])
    candidate_fit=costs(points,plane_flags,sources[:4]).sort(0).values[:3].mean(0).masked_fill(~valid,float('inf'))
    best,which=candidate_fit.min(0); eligible=valid.any(0)&valid_old[0]&torch.isfinite(best)
    rows=torch.arange(n,device=device)
    proposed=raw_depth[which,rows]; chosen_slopes=raw_slopes[which,rows]
    safe_new=torch.where(eligible,proposed,old); safe_slopes=torch.where(eligible[:,None],chosen_slopes,incumbent)
    points_new,valid_new=plane_support(uv,safe_new[None],safe_slopes,views[index],k,near,far)
    fit_old=costs(points_old,valid_old,sources[:4]).sort(0).values[:3].mean(0)[0]
    held=costs(torch.cat([points_old,points_new]),torch.cat([valid_old,valid_new]),sources[4:])
    result,accepted=accept_rgb(old,proposed,fit_old,best,held[:,0],held[:,1],eligible,True)
    empty=torch.full_like(old,float('nan'))
    diag={'accepted':accepted,'eligible':eligible,'incumbent_slopes':incumbent,'incumbent_fallback':fallback,
        **hyp,'candidate_fit':candidate_fit,
        'selected_hypothesis':torch.where(eligible,which,torch.full_like(which,-1)),
        'selected_source':torch.where(eligible,torch.tensor(sources[:4],device=device)[which],torch.full_like(which,-1)),
        'proposed':torch.where(eligible,proposed,empty),'fit_old':fit_old,'fit_new':best,
        'held_old':held[:,0],'held_new':held[:,1]}
    return result,diag
