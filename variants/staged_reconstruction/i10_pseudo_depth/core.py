"""Pure local calibration and motion tests; no GT or appearance classification."""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError
from ..pretrained_depth.core import affine_fit
from ..i10_tracking.core import sample


def in_hull(points, anchors):
    points=np.asarray(points,float).reshape(-1,2);anchors=np.asarray(anchors,float).reshape(-1,2)
    if len(anchors)<3:return np.zeros(len(points),bool)
    try:return Delaunay(anchors).find_simplex(points)>=0
    except QhullError:return np.zeros(len(points),bool)


def depth_at(raw, uv, coefficients):
    values=sample(raw, np.asarray(uv,np.float32)).astype(float)
    inverse=coefficients[0]*values+coefficients[1]
    good=np.isfinite(inverse)&(inverse>0)
    z=np.full(len(values),np.nan);z[good]=1/inverse[good]
    return z


def calibrate(raw, anchors, config):
    # Same spatial cell never appears on both sides, even for overlapping tracks.
    selected={}
    for q in sorted(anchors,key=lambda q:q['id']):
        key=tuple((np.asarray(q['uv'])//config['deduplicate_px']).astype(int))
        if key not in selected:selected[key]=q
    rows=list(selected.values());uv=np.array([q['uv'] for q in rows],float).reshape(-1,2)
    z=np.array([q['depth'] for q in rows],float)
    held=((uv[:,0]//8+uv[:,1]//8)%3)==0
    fit=np.flatnonzero(~held);check=np.flatnonzero(held)
    result=dict(passed=False,reason='insufficient_anchors',anchors=len(rows),
                fit_ids=fit.tolist(),check_ids=check.tolist(),anchor_ids=[q['id'] for q in rows],
                uv=uv.tolist(),depth=z.tolist(),coefficients=None)
    if len(fit)<config['minimum_fit'] or len(check)<config['minimum_check']:return result
    extent=np.ptp(uv[fit],axis=0)/np.array([raw.shape[1],raw.shape[0]])
    spread=float(np.ptp(1/z[fit]));result.update(extent=extent.tolist(),inverse_spread=spread)
    if (extent<config['minimum_extent']).any() or spread<config['minimum_inverse_spread']:
        result['reason']='degenerate_geometry';return result
    x=sample(raw,uv).astype(float);coeff=affine_fit(x[fit],1/z[fit])
    if coeff is None:result['reason']='nonpositive_or_degenerate_fit';return result
    result['coefficients']=list(coeff)
    predicted=depth_at(raw,uv[check],coeff)
    if not np.isfinite(predicted).all():result['reason']='invalid_check_depth';return result
    error=np.abs(predicted-z[check])/z[check]
    result.update(check_median_relative=float(np.median(error)),check_P90_relative=float(np.quantile(error,.9)),
                  check_inside_fit_hull=int(in_hull(uv[check],uv[fit]).sum()))
    # Disjoint spatial checkerboard partitions of FIT anchors only.
    halves=((uv[fit,0]//8)%2).astype(int)
    independent=[]
    for half in [0,1]:
        ids=fit[halves==half]
        if len(ids)<config['minimum_split_fit']:
            result['reason']='insufficient_split_fit';return result
        c=affine_fit(x[ids],1/z[ids])
        if c is None:result['reason']='degenerate_split_fit';return result
        independent.append(c)
    a,b=[depth_at(raw,uv[check],c) for c in independent]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        result['reason']='invalid_split_check';return result
    stability=np.abs(a-b)/z[check]
    result.update(split_coefficients=[list(c) for c in independent],
                  split_median_relative=float(np.median(stability)),split_P90_relative=float(np.quantile(stability,.9)))
    good=(np.median(error)<=config['relative_median_max'] and np.quantile(error,.9)<=config['relative_P90_max']
          and np.median(stability)<=config['relative_median_max'] and np.quantile(stability,.9)<=config['relative_P90_max'])
    result.update(passed=bool(good),reason='passed_local_anchor_check' if good else 'anchor_or_scale_disagreement')
    return result


def boundary(raw, uv, coefficients, radius=2):
    uv=np.asarray(uv,float).reshape(-1,2)
    offsets=np.array([(x,y) for y in range(-radius,radius+1) for x in range(-radius,radius+1)])
    pixels=uv[:,None]+offsets;h,w=raw.shape
    inside=((pixels[...,0]>=0)&(pixels[...,0]<=w-1)&(pixels[...,1]>=0)&(pixels[...,1]<=h-1)).all(1)
    z=depth_at(raw,pixels.reshape(-1,2),coefficients).reshape(len(uv),-1)
    good=inside&np.isfinite(z).all(1)
    contrast=np.full(len(uv),np.nan)
    contrast[good]=(z[good].max(1)-z[good].min(1))/np.median(z[good],axis=1)
    return contrast,good


def backproject(uv, z, views, k):
    uv=np.asarray(uv,float);z=np.asarray(z,float)
    ray=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(k).T
    camera=ray*z[:,None];inverse=np.linalg.inv(views)
    return np.einsum('tij,tj->ti',inverse[:,:3,:3],camera)+inverse[:,:3,3]


def motion_check(prior_world, parent_world, valid, depths, config):
    prior_world=np.asarray(prior_world,float);parent_world=np.asarray(parent_world,float)
    valid=np.asarray(valid,bool)&np.isfinite(prior_world).all(1)&np.isfinite(parent_world).all(1)
    fit=np.flatnonzero(valid&(np.arange(len(valid))%2==0));held=np.flatnonzero(valid&(np.arange(len(valid))%2==1))
    result=dict(passed=False,reason='insufficient_observations',valid=int(valid.sum()),fit_count=len(fit),held_count=len(held))
    if len(fit)<config['minimum_valid_fit'] or len(held)<config['minimum_valid_check']:return result
    offset=prior_world-parent_world;reference=np.median(offset[fit],axis=0)
    corrected=np.linalg.norm(offset[held]-reference,axis=1)/depths[held]
    camera_only=np.linalg.norm(prior_world[held]-np.median(prior_world[fit],axis=0),axis=1)/depths[held]
    result.update(fit_offset=reference.tolist(),motion_median_relative=float(np.median(corrected)),
                  motion_P90_relative=float(np.quantile(corrected,.9)),camera_only_median_relative=float(np.median(camera_only)),
                  held_ids=held.tolist(),fit_ids=fit.tolist())
    good=np.median(corrected)<=config['motion_median_max'] and np.quantile(corrected,.9)<=config['motion_P90_max']
    result.update(passed=bool(good),reason='passed_model_conditioned_motion_check' if good else 'motion_disagreement')
    return result


def weak_depth(base, current_depth=None, target_depth=None, confidence=None, enabled=False, weight=.001):
    if not enabled:return base
    import torch
    if current_depth.numel()==0:return base+current_depth.sum()*0
    assert current_depth.shape==target_depth.shape==confidence.shape
    assert bool(torch.isfinite(target_depth).all() and (target_depth>0).all())
    assert bool(torch.isfinite(current_depth).all() and (current_depth>0).all())
    error=torch.log(current_depth/target_depth.detach())
    energy=torch.sqrt(1+(error/.05).square())-1
    return base+weight*(confidence.detach()*energy).mean()


def transition(previous, evidence):
    state=dict(previous or dict(status='suspended',failures=0,passes=0))
    if evidence=='unknown':state.update(status='suspended',passes=0);return state
    if evidence=='contradiction':
        state.update(failures=state['failures']+1,passes=0)
        state['status']='revoked' if state['failures']>=2 else 'suspended'
    elif evidence=='pass':
        state.update(passes=state['passes']+1,failures=0)
        state['status']='active' if state['passes']>=2 else 'suspended'
    else:raise ValueError('Unknown evidence')
    return state
