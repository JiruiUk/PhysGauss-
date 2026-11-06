"""RGB hypotheses, group-held calibration and fixed temporal affine fit."""
import hashlib
from collections import Counter, defaultdict
import cv2
import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError
from scipy import sparse
from scipy.optimize import least_squares
from ..i10_tracking.core import grayscale, qualify, stationary_test, project_world, sample
from ..pretrained_depth.core import affine_fit


def features(image, p):
    g=np.uint8(np.clip(grayscale(image),0,255));h,w=g.shape;points=[]
    for y in range(p['tiles']):
        for x in range(p['tiles']):
            x0,x1=x*w//p['tiles'],(x+1)*w//p['tiles']
            y0,y1=y*h//p['tiles'],(y+1)*h//p['tiles']
            found=cv2.goodFeaturesToTrack(g[y0:y1,x0:x1],p['corners_per_tile'],p['quality'],
                p['minimum_distance'],blockSize=p['block_size'])
            if found is not None:
                for q in found[:,0]:points.append(q+[x0,y0])
    return np.asarray(points,np.float32).reshape(-1,2)


def flow(images, seeds, p):
    gs=[np.uint8(np.clip(grayscale(q),0,255)) for q in images]
    n=len(seeds);forward=np.empty((len(gs),n,2),np.float32);forward[0]=seeds
    status=np.ones((len(gs),n),bool)
    for t in range(1,len(gs)):
        if not n:forward[t]=seeds;continue
        q,s,_=cv2.calcOpticalFlowPyrLK(gs[t-1],gs[t],forward[t-1][:,None],None,
            winSize=(p['window'],)*2,maxLevel=p['pyramid_levels'],
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,p['iterations'],p['epsilon']))
        forward[t]=q[:,0];status[t]=s[:,0].astype(bool)
    return forward,status


def extract(images, views, k, start, p, original):
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    seed=features(images[0],p['lk']);f,v=flow(images,seed,p['lk'])
    r,rv=flow(images[::-1],f[-1],p['lk']);r=r[::-1].copy();rv=rv[::-1].copy()
    # The first observation of each LK pass is prescribed, as with old queries.
    good,diagnostics=qualify(f,r,v,rv,[grayscale(q) for q in images],original)
    rows=[]
    for j in np.flatnonzero(good):
        result=stationary_test(f[:,j],views,k,original)
        rows.append(dict(id=f'lk_{start:03d}_{j:03d}',start=start,uv=f[:,j].tolist(),stationarity=result))
    return dict(seeds=seed,forward=f,reverse=r,visible=v,reverse_visible=rv,accepted=good),dict(
        start=start,seeds=len(seed),qualified=int(good.sum()),rows=rows,diagnostics=diagnostics)


def group_tracks(rows, p):
    """Merge duplicate measured tracks; never infer an object identity."""
    rows=sorted(rows,key=lambda q:q['id']);parent=list(range(len(rows)))
    byframe=defaultdict(list)
    for j,q in enumerate(rows):
        for t,uv in enumerate(q['uv']):byframe[q['start']+t].append((j,uv))
    close=Counter()
    for values in byframe.values():
        if len(values)<2:continue
        pairs=cKDTree(np.array([q[1] for q in values])).query_pairs(p['maximum_separation_px'])
        for a,b in pairs:close[tuple(sorted((values[a][0],values[b][0])))]+=1
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for (i,j),n in sorted(close.items()):
        a,b=rows[i],rows[j]
        overlap=min(a['start']+len(a['uv']),b['start']+len(b['uv']))-max(a['start'],b['start'])
        if overlap>=p['minimum_overlap'] and n==overlap:
            x,y=root(i),root(j);parent[max(x,y)]=min(x,y)
    groups={q['id']:rows[root(i)]['id'] for i,q in enumerate(rows)}
    return groups


def partition(group):
    n=int(hashlib.sha256(group.encode()).hexdigest()[:16],16)
    return 'check' if n%3==0 else ('fit0' if (n//3)%2==0 else 'fit1')


def anchor_bank(rows, views, k, p):
    groups=group_tracks(rows,p);bank=[[] for _ in range(len(views))]
    for q in sorted(rows,key=lambda q:q['id']):
        start=q['start'];point=q['stationarity']['stationary_hypothesis_xyz']
        _,z=project_world(np.array(point)[None],views[start:start+len(q['uv'])],k)
        for t,uv in enumerate(q['uv']):bank[start+t].append(dict(id=q['id'],group=groups[q['id']],
            split=partition(groups[q['id']]),uv=uv,depth=float(z[t,0])))
    counts=[];clean=[]
    for rows in bank:
        cells=defaultdict(list)
        for q in rows:cells[tuple((np.array(q['uv'])//p['deduplicate_px']).astype(int))].append(q)
        selected=[]
        for _,values in sorted(cells.items()):
            held=[q for q in values if q['split']=='check']
            selected.append(sorted(held or values,key=lambda q:q['id'])[0])
        clean.append(selected);counts.append(dict(before=len(rows),after=len(selected),removed=len(rows)-len(selected)))
    return clean,dict(groups=groups,group_sizes=dict(Counter(groups.values())),deduplication=counts)


def normalize(raw):
    median=float(np.median(raw));spread=float(np.quantile(raw,.9)-np.quantile(raw,.1))
    return median,spread


def joint_fit(bank, norms, raw, p, half=None, enabled=False, original=None):
    if not enabled:return original
    n=len(bank);fi=[];xs=[];ys=[]
    for i,rows in enumerate(bank):
        chosen=[q for q in rows if q['split']!='check' and (half is None or q['split']==f'fit{half}')]
        med,spread=norms[i]
        if not chosen or spread<=1e-12:continue
        x=(sample(raw[i],np.array([q['uv'] for q in chosen]))-med)/spread
        y=1/np.array([q['depth'] for q in chosen])
        fi.extend([i]*len(y));xs.extend(x);ys.extend(y)
    if not ys:return dict(success=False,reason='no_fit_anchors',coefficients=None)
    fi=np.array(fi);x=np.array(xs,float);y=np.array(ys);scale=float(np.median(y))
    count=np.bincount(fi,minlength=n);weight=1/count[fi]
    data=sparse.coo_matrix((np.r_[x/y,1/y],
        (np.r_[np.arange(len(y)),np.arange(len(y))],np.r_[fi*2,fi*2+1])),shape=(len(y),n*2)).tocsr()
    d2=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n))
    smooth=sparse.kron(d2,sparse.eye(2))*np.sqrt(p['smoothness'])/scale
    ridge=sparse.eye(n*2)*np.sqrt(p['ridge'])/scale
    matrix=sparse.vstack([data,smooth,ridge]).tocsr()
    target=np.r_[np.ones(len(y)),np.zeros(smooth.shape[0]+ridge.shape[0])]
    initial=np.tile([scale*.1,scale],n)
    # A fixed neutral positive initialization, never a held-driven warm start.
    lower=np.tile([p['minimum_slope'],-np.inf],n)
    nd=len(y)
    def loss(z):
        result=np.vstack([z,np.ones_like(z),np.zeros_like(z)])
        take=np.flatnonzero(z[:nd]>1)
        result[0,take]=2*np.sqrt(z[take])-1
        result[1,take]=1/np.sqrt(z[take]);result[2,take]=-.5/z[take]**1.5
        result[:,:nd]*=weight
        return result
    result=least_squares(lambda v:matrix@v-target,initial,jac=lambda v:matrix.copy(),
        bounds=(lower,np.full(2*n,np.inf)),loss=loss,f_scale=p['huber_scale'],
        max_nfev=p['maximum_iterations'],ftol=p['solver_tolerance'],
        xtol=p['solver_tolerance'],gtol=p['solver_tolerance'])
    normalized=result.x.reshape(n,2);coeff=[]
    for (a,b),(med,spread) in zip(normalized,norms):
        coeff.append([a/spread,b-a*med/spread] if spread>1e-12 else [np.nan,np.nan])
    return dict(success=bool(result.success),reason=str(result.message),coefficients=coeff,
        normalized_coefficients=normalized.tolist(),scale=scale,fit_observations=len(y),
        iterations=int(result.nfev),cost=float(result.cost),optimality=float(result.optimality),
        coefficients_model='positive normalized affine with fixed temporal second-difference prior')


def coverage(uv,shape):
    uv=np.asarray(uv,float).reshape(-1,2);h,w=shape
    occupied=len(set(map(tuple,(uv/np.array([w,h])*4).astype(int)))) if len(uv) else 0
    area=0.
    if len(uv)>=3:
        try:area=float(ConvexHull(uv).volume/(w*h))
        except QhullError:pass
    return dict(occupied_4x4_cells=occupied,hull_area_fraction=area)


def check_frame(raw, rows, coefficients, split_coefficients, p, solved=True):
    fit=[q for q in rows if q['split']!='check'];check=[q for q in rows if q['split']=='check']
    uv=np.array([q['uv'] for q in rows]).reshape(-1,2)
    result=dict(passed=False,reason='insufficient_anchors',anchors=len(rows),
        uv=uv.tolist(),depth=[q['depth'] for q in rows],anchor_ids=[q['id'] for q in rows],
        groups=[q['group'] for q in rows],splits=[q['split'] for q in rows],
        fit_ids=[j for j,q in enumerate(rows) if q['split']!='check'],
        check_ids=[j for j,q in enumerate(rows) if q['split']=='check'],
        coefficients=coefficients,split_coefficients=split_coefficients,
        coverage=coverage([q['uv'] for q in fit],raw.shape))
    if len(fit)<p['minimum_fit'] or len(check)<p['minimum_check']:return result
    fu=np.array([q['uv'] for q in fit]);fz=np.array([q['depth'] for q in fit]);cu=np.array([q['uv'] for q in check]);cz=np.array([q['depth'] for q in check])
    extent=np.ptp(fu,axis=0)/np.array([raw.shape[1],raw.shape[0]])
    if (extent<p['minimum_extent']).any() or np.ptp(1/fz)<p['minimum_inverse_spread']:
        result['reason']='degenerate_geometry';return result
    if min(sum(q['split']==s for q in fit) for s in ['fit0','fit1'])<p['minimum_split_fit']:
        result['reason']='insufficient_split_fit';return result
    if not solved or coefficients is None or any(c is None for c in split_coefficients):
        result['reason']='solver_or_fit_failed';return result
    from ..i10_pseudo_depth.core import depth_at
    z=depth_at(raw,cu,coefficients);a,b=[depth_at(raw,cu,c) for c in split_coefficients]
    if not np.isfinite(np.r_[z,a,b]).all():result['reason']='invalid_check_depth';return result
    error=np.abs(z-cz)/cz;stability=np.abs(a-b)/cz
    result.update(check_median_relative=float(np.median(error)),check_P90_relative=float(np.quantile(error,.9)),
        split_median_relative=float(np.median(stability)),split_P90_relative=float(np.quantile(stability,.9)))
    good=(np.median(error)<=p['relative_median_max'] and np.quantile(error,.9)<=p['relative_P90_max']
          and np.median(stability)<=p['relative_median_max'] and np.quantile(stability,.9)<=p['relative_P90_max'])
    result.update(passed=bool(good),reason='passed_local_anchor_check' if good else 'anchor_or_scale_disagreement')
    return result


def framewise(raw,rows,p):
    co=[]
    for half in [None,0,1]:
        selected=[q for q in rows if q['split']!='check' and (half is None or q['split']==f'fit{half}')]
        co.append(affine_fit(sample(raw,np.array([q['uv'] for q in selected]).reshape(-1,2)),1/np.array([q['depth'] for q in selected])) if len(selected)>=2 else None)
    return check_frame(raw,rows,co[0],co[1:],p)
