"""Motion grouping without a requested number of groups or a K-means stage."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import HDBSCAN
from threadpoolctl import threadpool_limits

from variants.object_centers.motion_grouping import motion_features,stable_ids


def validate(poses,times,eligible,weights):
    if eligible.shape!=(poses.shape[1],) or weights.shape!=eligible.shape:
        raise ValueError('Aligned eligibility and weights required')
    selected=np.flatnonzero(eligible)
    if not np.isfinite(weights).all() or (weights[selected]<=0).any():
        raise ValueError('Positive finite center weights required')
    return selected


def density_motion(poses,times,eligible,weights,mode,min_cluster_size=128,min_samples=16):
    selected=validate(poses,times,eligible,weights)
    if min_cluster_size<2 or min_samples<1:raise ValueError('Invalid density parameters')
    labels=np.full(poses.shape[1],-1,dtype=np.int32);confidence=np.zeros(len(labels),dtype=np.float32)
    if len(selected)>=max(min_cluster_size,min_samples):
        features=motion_features(poses[:,selected],times,mode)
        # Tree-based exact implementation avoids materializing an N*N matrix.
        # Allow one group: do not implicitly require at least two objects.
        with threadpool_limits(limits=1):
            model=HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,
                metric='euclidean',algorithm='kd_tree',leaf_size=80,n_jobs=1,
                cluster_selection_method='eom',allow_single_cluster=True,copy=True)
            labels[selected]=model.fit_predict(features)
        confidence[selected]=model.probabilities_
        labels=stable_ids(labels,poses[0],weights)
    stats={'algorithm':'sklearn.HDBSCAN','requested_group_count':None,'actual_groups':int(labels.max()+1),
        'min_cluster_size':min_cluster_size,'min_samples':min_samples,'selection':'eom',
        'allow_single_cluster':True,'eligible_noise_kernels':int((labels[selected]<0).sum()),
        'sample_weights_used_for_density':False,'center_weights_used':True,
        'confidence_mean_assigned':float(confidence[labels>=0].mean()) if (labels>=0).any() else None}
    return labels,stats,confidence


def spatial_motion_graph(poses,times,eligible,weights,min_cluster_size=128,neighbors=16,
                         radius_factor=2.5,radius_cap=.08,persistence=.8,motion_radius=.15):
    """Direct spatial graph with a motion-distance gate; no base clustering.

    Edges must satisfy normalized displacement+velocity distance AND persistent
    spatial proximity. Connected component counts emerge from the graph. Small
    components remain unknown; neither K nor a density-derived base ID is used.
    """
    selected=validate(poses,times,eligible,weights);n=len(selected)
    if min_cluster_size<1 or neighbors<1 or min(radius_factor,radius_cap,motion_radius)<=0 or not 0<persistence<=1:
        raise ValueError('Invalid graph parameters')
    labels=np.full(poses.shape[1],-1,dtype=np.int32)
    stats={'algorithm':'direct motion/spatial graph','requested_group_count':None,
        'actual_groups':0,'min_cluster_size':min_cluster_size,'neighbors':neighbors,
        'radius_factor':radius_factor,'radius_cap_scene_units':radius_cap,
        'persistence_sample_fraction':persistence,'motion_radius_normalized_features':motion_radius,
        'candidate_edges':0,'motion_compatible_edges':0,'persistent_edges':0,'raw_components':0}
    if n<min_cluster_size:return labels,stats
    features=motion_features(poses[:,selected],times,'displacement_velocity')
    distances,nearest=cKDTree(poses[0,selected]).query(poses[0,selected],k=min(neighbors+1,n),workers=1)
    if distances.ndim==1:distances=distances[:,None];nearest=nearest[:,None]
    local=np.maximum(distances[:,-1],1e-6)
    i=np.repeat(np.arange(n),nearest.shape[1]-1);j=nearest[:,1:].reshape(-1)
    pairs=np.unique(np.sort(np.stack([i,j],axis=1),axis=1),axis=0)
    pairs=pairs[pairs[:,0]!=pairs[:,1]];i,j=pairs.T
    stats['candidate_edges']=len(i)
    compatible=np.linalg.norm(features[i]-features[j],axis=1)<=motion_radius
    i,j=i[compatible],j[compatible];stats['motion_compatible_edges']=len(i)
    threshold=np.minimum(radius_cap,radius_factor*np.minimum(local[i],local[j]))
    hits=np.zeros(len(i),dtype=np.int32)
    for pose in poses:hits+=np.linalg.norm(pose[selected[i]]-pose[selected[j]],axis=-1)<=threshold
    keep=hits>=int(np.ceil(persistence*len(poses)))
    graph=coo_matrix((np.ones(keep.sum()*2),(np.r_[i[keep],j[keep]],np.r_[j[keep],i[keep]])),shape=(n,n)).tocsr()
    count,component=connected_components(graph,directed=False)
    accepted=np.bincount(component,minlength=count)>=min_cluster_size
    good=accepted[component];labels[selected[good]]=component[good]
    labels=stable_ids(labels,poses[0],weights)
    stats.update(actual_groups=int(labels.max()+1),persistent_edges=int(keep.sum()),raw_components=int(count),
        eligible_noise_kernels=int((~good).sum()))
    return labels,stats
