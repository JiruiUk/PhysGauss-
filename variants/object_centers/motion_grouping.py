"""Fixed motion-only candidates; outputs are not physical instance identities."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits


def motion_features(poses,times,mode):
    poses=np.asarray(poses,dtype=np.float64);times=np.asarray(times,dtype=np.float64)
    if poses.ndim!=3 or poses.shape[2]!=3 or poses.shape[0]!=len(times) or len(times)<2:
        raise ValueError('Expected poses[T,N,3] and times[T]')
    if not np.isfinite(poses).all() or not np.isfinite(times).all() or not (np.diff(times)>0).all():
        raise ValueError('Finite trajectories and increasing time knots required')
    if mode not in ('displacement','displacement_velocity'):raise ValueError('Unknown motion feature mode')
    dt=np.diff(times);dt=dt/(times[-1]-times[0])
    # Quadrature prevents irregular frame spacing from changing sample importance.
    quadrature=np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2]
    delta=poses-poses[0]
    amplitudes=np.sqrt(np.sum(delta**2*quadrature[:,None,None],axis=(0,2)))
    displacement_scale=max(float(np.percentile(amplitudes,95)),1e-6)
    blocks=[(delta*np.sqrt(quadrature)[:,None,None]/displacement_scale).transpose(1,0,2).reshape(len(poses[0]),-1)]
    if mode=='displacement_velocity':
        # Derived from the deformation trajectory, not PIDG's velocity head.
        velocity=np.diff(poses,axis=0)/np.diff(times)[:,None,None]
        speeds=np.sqrt(np.sum(velocity**2*dt[:,None,None],axis=(0,2)))
        velocity_scale=max(float(np.percentile(speeds,95)),1e-6)
        blocks.append((velocity*np.sqrt(dt)[:,None,None]/velocity_scale).transpose(1,0,2).reshape(len(poses[0]),-1))
    return np.concatenate(blocks,axis=1).astype(np.float32)


def stable_ids(labels,reference,weights):
    present=np.unique(labels[labels>=0])
    centers=np.array([np.average(reference[labels==g],axis=0,weights=weights[labels==g]) for g in present])
    result=np.full(len(labels),-1,dtype=np.int32)
    if not len(present):return result
    for new,old in enumerate(present[np.lexsort((centers[:,0],-centers[:,1]))]):result[labels==old]=new
    return result


def cluster_motion(poses,times,eligible,weights,groups,mode):
    selected=np.flatnonzero(eligible)
    if len(selected)<groups or groups<1:raise ValueError('Not enough eligible kernels')
    if not np.isfinite(weights).all() or (weights[selected]<=0).any():raise ValueError('Positive finite weights required')
    features=motion_features(poses[:,selected],times,mode)
    with threadpool_limits(limits=1):
        model=MiniBatchKMeans(n_clusters=groups,random_state=0,n_init=3,batch_size=4096,
                             max_iter=160,reassignment_ratio=0.)
        model.fit(features,sample_weight=weights[selected]);assignment=model.predict(features)
    labels=np.full(poses.shape[1],-1,dtype=np.int32);labels[selected]=assignment
    return stable_ids(labels,poses[0],weights)


def persistent_components(poses,base_labels,weights,neighbors=16,radius_factor=2.5,
                          radius_cap=.08,persistence=.8,min_points=128):
    """Refine motion clusters with a spatial graph persistent across sampled times.

    Candidate edges are reference-pose kNN, restricted to the same base motion
    cluster. Their adaptive distance threshold is capped in scene units. An edge
    survives only if it stays near in >=persistence of sampled training poses.
    Tiny components remain unknown. Persistence is a sample fraction, not a
    continuous-time collision guarantee, and kNN connectivity is only a heuristic.
    """
    if not 0<persistence<=1 or radius_cap<=0 or radius_factor<=0 or neighbors<1 or min_points<1:
        raise ValueError('Invalid connectivity parameters')
    selected=np.flatnonzero(base_labels>=0);n=len(selected)
    result=np.full(len(base_labels),-1,dtype=np.int32)
    if not n:return result,{'kept_components':0,'discarded_kernels':0,'candidate_edges':0,'persistent_edges':0}
    ref=poses[0,selected]
    distance,nearest=cKDTree(ref).query(ref,k=min(neighbors+1,n),workers=1)
    if distance.ndim==1:distance=distance[:,None];nearest=nearest[:,None]
    # The furthest available neighbor estimates local sampling density.
    local=np.maximum(distance[:,-1],1e-6)
    i=np.repeat(np.arange(n),nearest.shape[1]-1);j=nearest[:,1:].reshape(-1)
    same=base_labels[selected[i]]==base_labels[selected[j]]
    i,j=i[same],j[same]
    pairs=np.unique(np.sort(np.stack([i,j],axis=1),axis=1),axis=0)
    i,j=pairs.T
    threshold=np.minimum(radius_cap,radius_factor*np.minimum(local[i],local[j]))
    hits=np.zeros(len(i),dtype=np.int32)
    for pose in poses:
        hits+=np.linalg.norm(pose[selected[i]]-pose[selected[j]],axis=-1)<=threshold
    keep=hits>=int(np.ceil(persistence*len(poses)))
    graph=coo_matrix((np.ones(keep.sum()*2),
        (np.r_[i[keep],j[keep]],np.r_[j[keep],i[keep]])),shape=(n,n)).tocsr()
    count,components=connected_components(graph,directed=False)
    sizes=np.bincount(components,minlength=count);accepted=sizes>=min_points
    surviving=accepted[components];result[selected[surviving]]=components[surviving]
    result=stable_ids(result,poses[0],weights)
    stats={'kept_components':int(accepted.sum()),'discarded_kernels':int((~surviving).sum()),
           'candidate_edges':len(i),'persistent_edges':int(keep.sum()),'raw_components':int(count),
           'neighbors':neighbors,'radius_factor':radius_factor,'radius_cap_scene_units':radius_cap,
           'persistence_sample_fraction':persistence,'min_component_kernels':min_points}
    return result,stats
