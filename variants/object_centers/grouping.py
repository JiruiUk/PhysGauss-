"""One-time unsupervised candidates; never semantic or material ground truth."""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits


def cluster_candidates(canonical,poses,colors,eligible,weights,groups,mode='trajectory'):
    if mode not in ('canonical','trajectory','motion_dominant'):raise ValueError('Unknown candidate feature mode')
    indices=np.flatnonzero(eligible)
    if len(indices)<groups:raise ValueError('Not enough supported kernels')
    reference=canonical[indices] if mode=='canonical' else poses[0,indices]
    scale=max(float(np.linalg.norm(np.percentile(reference,95,axis=0)-np.percentile(reference,5,axis=0))),1e-5)
    features=[reference/scale*(.15 if mode=='motion_dominant' else 1.)]
    if mode in ('trajectory','motion_dominant'):
        delta=poses[:,indices]-poses[0,indices][None]
        factor=4 if mode=='motion_dominant' else 2
        features += [factor*delta.transpose(1,0,2).reshape(len(indices),-1)/(scale*np.sqrt(len(poses))),colors[indices]*.75]
    data=np.concatenate(features,axis=-1).astype(np.float32)
    with threadpool_limits(limits=1):
        model=MiniBatchKMeans(n_clusters=groups,random_state=0,n_init=3,batch_size=4096,
            max_iter=160,reassignment_ratio=0.)
        model.fit(data,sample_weight=weights[indices]);assignment=model.predict(data)
    # Stable palette ordering for this candidate, using its reference centers.
    centers=np.array([np.average(reference[assignment==k],axis=0,weights=weights[indices][assignment==k]) for k in range(groups)])
    order=np.lexsort((centers[:,0],-centers[:,1]));inverse=np.empty(groups,dtype=np.int32);inverse[order]=np.arange(groups)
    labels=np.full(len(canonical),-1,dtype=np.int32);labels[indices]=inverse[assignment]
    return labels
