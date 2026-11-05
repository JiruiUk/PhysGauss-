"""No semantic masks: image consistency, stationary compatibility and 3D binding."""
import cv2
import numpy as np
from scipy.optimize import least_squares


def grayscale(rgb):
    return np.asarray(rgb, np.float32).mean(-1)


def seeds(gray, config):
    gray = np.asarray(gray, np.float32)
    radius = config['patch_radius']; side = 2*radius+1
    mean = cv2.boxFilter(gray, -1, (side, side), normalize=True)
    var = cv2.boxFilter(gray*gray, -1, (side, side), normalize=True)-mean*mean
    std = np.sqrt(np.maximum(var, 0))
    h, w = gray.shape; margin = config['margin_px']; chosen = []
    tiles=config['seed_tiles']
    for ty in range(tiles):
        for tx in range(tiles):
            y, x = np.mgrid[max(margin,ty*h//tiles):min(h-margin,(ty+1)*h//tiles),
                           max(margin,tx*w//tiles):min(w-margin,(tx+1)*w//tiles)]
            # Lexicographic tie breaks preserve deterministic, full-image seeds.
            good = std[y, x] >= config['patch_std_min']; x=x[good]; y=y[good]
            order = np.lexsort((x,y,-std[y,x])); added=0
            for index in order:
                point=np.array([x[index],y[index]],np.float32)
                if chosen and np.min(np.linalg.norm(np.array(chosen)-point,axis=1)) < config['seed_distance']:
                    continue
                chosen.append(point); added+=1
                if added == config['seeds_per_tile']: break
    return np.asarray(chosen,np.float32).reshape(-1,2)


def sample(field, points):
    points=np.asarray(points,np.float32)
    return cv2.remap(np.asarray(field,np.float32),points[:,0:1],points[:,1:2],
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(len(points))


def patch_ncc(a, b, x, y, radius):
    offsets=np.array([(i,j) for j in range(-radius,radius+1) for i in range(-radius,radius+1)],np.float32)
    shape=(len(x),len(offsets))
    p=sample(a,(x[:,None]+offsets).reshape(-1,2)).reshape(shape)
    q=sample(b,(y[:,None]+offsets).reshape(-1,2)).reshape(shape)
    p-=p.mean(1,keepdims=True);q-=q.mean(1,keepdims=True)
    denom=np.linalg.norm(p,axis=1)*np.linalg.norm(q,axis=1)
    score=np.divide((p*q).sum(1),denom,out=np.zeros(len(p),np.float32),where=denom>1e-8)
    return score,np.sqrt((p*p).mean(1)),np.sqrt((q*q).mean(1))


def qualify(forward, reverse, visible, reverse_visible, grays, config):
    f,r=np.asarray(forward),np.asarray(reverse)
    assert f.shape==r.shape and len(f)==config['clip_length']
    finite=np.isfinite(f).all((0,2)) & np.isfinite(r).all((0,2))
    safe=np.nan_to_num(f,nan=-100.,posinf=-100.,neginf=-100.)
    h,w=grays[0].shape;m=config['margin_px']
    inside=((f[...,0]>=m)&(f[...,0]<=w-1-m)&(f[...,1]>=m)&(f[...,1]<=h-1-m)).all(0)
    closure=np.linalg.norm(r[0]-f[0],axis=-1)
    # Forward[0] and reverse[-1] were forced by the official predictor.
    agreement=np.linalg.norm(r[1:-1]-f[1:-1],axis=-1).max(0)
    scores=[];textures=[]
    for t in range(1,len(f)):
        score,a,b=patch_ncc(grays[t-1],grays[t],safe[t-1],safe[t],config['patch_radius'])
        scores.append(score);textures.append(np.minimum(a,b))
    ncc=np.min(scores,axis=0);texture=np.min(textures,axis=0)
    checks=dict(finite=finite,inside=inside,
        visible=np.asarray(visible[1:],bool).all(0)&np.asarray(reverse_visible[:-1],bool).all(0),
        closure=closure<=config['closure_max_px'],agreement=agreement<=config['direction_max_px'],
        texture=texture>=config['patch_std_min'],ncc=ncc>=config['ncc_min'])
    good=np.logical_and.reduce(list(checks.values()))
    return good,dict(checks={k:v.tolist() for k,v in checks.items()},closure_px=closure.tolist(),
        agreement_px=agreement.tolist(),minimum_ncc=ncc.tolist(),minimum_std=texture.tolist())


def project_world(xyz, views, k):
    xyz=np.asarray(xyz,np.float64);views=np.asarray(views,np.float64)
    if xyz.ndim==2:xyz=np.broadcast_to(xyz,(len(views),)+xyz.shape)
    p=np.einsum('tij,tnj->tni',views[:,:3,:3],xyz)+views[:,:3,3,None].transpose(0,2,1)
    z=p[...,2];q=p@np.asarray(k).T
    with np.errstate(divide='ignore',invalid='ignore'):uv=q[...,:2]/q[...,2:3]
    return uv,z


def stationary_test(uv, views, k, config):
    """A fixed-world *hypothesis test*, never moving-object depth ground truth."""
    uv=np.asarray(uv,np.float64);views=np.asarray(views,np.float64)
    fit=np.asarray(config['fit_observations']);held=np.asarray(config['held_observations'])
    inverse=np.linalg.inv(views);centers=inverse[:,:3,3]
    ray=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(k).T
    ray=np.einsum('tij,tj->ti',inverse[:,:3,:3],ray);ray/=np.linalg.norm(ray,axis=1,keepdims=True)
    orth=np.eye(3)[None]-ray[fit,:,None]*ray[fit,None,:]
    matrix=orth.sum(0);eigen=np.linalg.eigvalsh(matrix)
    ratio=float(eigen[0]/max(eigen[-1],1e-30))
    angle=float(np.degrees(np.arccos(np.clip((ray[fit]@ray[fit].T).min(),-1,1))))
    result=dict(label='unknown',eigenvalue_ratio=ratio,parallax_degrees=angle)
    if ratio<config['ray_matrix_min_eigenvalue_ratio'] or angle<config['minimum_parallax_degrees']:
        return dict(result,reason='degenerate_stationary_hypothesis')
    x=np.linalg.solve(matrix,np.einsum('tij,tj->i',orth,centers[fit]))
    def residual(point):
        predicted,_=project_world(point[None],views[fit],k)
        return (predicted[:,0]-uv[fit]).reshape(-1)
    solution=least_squares(residual,x,max_nfev=100,loss='linear')
    predicted,z=project_world(solution.x[None],views,k)
    if not np.isfinite(predicted).all() or (z<=config['minimum_positive_depth']).any():
        return dict(result,reason='nonpositive_or_nonfinite_stationary_hypothesis')
    errors=np.linalg.norm(predicted[:,0]-uv,axis=-1)
    result.update(fit_rms_px=float(np.sqrt(np.mean(errors[fit]**2))),
                  held_median_px=float(np.median(errors[held])),max_px=float(errors.max()),
                  stationary_hypothesis_xyz=solution.x.tolist(),solver_success=bool(solution.success))
    if not solution.success:return dict(result,reason='stationary_solver_failed')
    if errors.max()<=config['stationary_max_error_px']:result['label']='stationary_compatible'
    elif np.median(errors[held])>config['motion_candidate_held_median_min_px']:result['label']='motion_candidate'
    return result


def bind_track(uv, xyz, predicted, depth, config):
    """Fit chooses a persistent ID; held observations cannot choose another ID."""
    fit=config['fit_observations'];held=config['held_observations']
    errors=np.linalg.norm(predicted-np.asarray(uv)[:,None],axis=-1)
    rms=np.sqrt(np.mean(errors[fit]**2,axis=0))
    valid=np.isfinite(errors).all(0)&np.isfinite(xyz).all((0,2))&(depth>config['minimum_positive_depth']).all(0)
    score=np.where(valid,rms,np.inf)
    row=int(np.argmin(score))
    if not np.isfinite(score[row]):return dict(accepted=False,reason='no_valid_parent_trajectory')
    separated=np.linalg.norm(xyz[len(xyz)//2]-xyz[len(xyz)//2,row],axis=-1)>=config['binding_alternative_3D_distance']
    alternatives=score[separated&valid]
    alternative=float(alternatives.min()) if len(alternatives) else None
    margin=alternative-float(score[row]) if alternative is not None else None
    held_rms=float(np.sqrt(np.mean(errors[held,row]**2)))
    checks=dict(fit=score[row]<=config['binding_fit_rms_max_px'],
        held=held_rms<=config['binding_held_rms_max_px'],all=errors[:,row].max()<=config['binding_all_max_px'],
        unambiguous=margin is not None and margin>=config['binding_fit_rms_alternative_margin_min_px'])
    return dict(accepted=bool(all(checks.values())),row=row,fit_rms_px=float(score[row]),held_rms_px=held_rms,
        max_px=float(errors[:,row].max()),alternative_fit_rms_px=alternative,margin_px=margin,
        checks={k:bool(v) for k,v in checks.items()},model_conditioned_world_xyz=xyz[:,row].tolist(),
        world_depth_is_independent_GT=False)


def tracking_objective(base_loss, world, views, k, observations, enabled=False, weight=0.01):
    """Optional robust reprojection of persistent 3D points, default identity."""
    if not enabled:return base_loss
    import torch
    camera=torch.einsum('tij,tnj->tni',views[:,:3,:3],world)+views[:,:3,3,None].transpose(1,2)
    q=camera@k.T
    assert bool((camera[...,2]>.01).all()),'Invalid binding must not become a target'
    predicted=q[...,:2]/q[...,2:3]
    error=torch.linalg.vector_norm(predicted-observations,dim=-1)
    return base_loss+weight*(torch.sqrt(1+error.square())-1).mean()
