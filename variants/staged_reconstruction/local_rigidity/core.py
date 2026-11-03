"""Rotation-aware canonical neighborhood coherence; no material labels."""
import numpy as np
import torch
from scipy.spatial import cKDTree


def canonical_graph(points,k=20,radius=.08):
    points=np.asarray(points,np.float32)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all(): raise ValueError('Finite Nx3 required')
    if k<1 or radius<=0 or len(points)<2: raise ValueError('Invalid graph input')
    distance,index=cKDTree(points).query(points,k=min(k+1,len(points)),workers=1)
    source=np.repeat(np.arange(len(points)),index.shape[1]); target=index.reshape(-1); distance=distance.reshape(-1)
    keep=(source!=target)&(distance>1e-6)&(distance<=radius)
    edges=np.stack([source[keep],target[keep]],axis=-1)
    rest=distance[keep].astype(np.float32)
    if not len(edges): raise ValueError('No supported edges')
    return edges,rest,np.exp(-.5*(rest/(radius/2))**2).astype(np.float32)


def quaternion_matrix(quaternion):
    q=torch.nn.functional.normalize(quaternion,dim=-1,eps=1e-12)
    r,i,j,k=q.unbind(-1)
    return torch.stack([1-2*(j*j+k*k),2*(i*j-k*r),2*(i*k+j*r),
                        2*(i*j+k*r),1-2*(i*i+k*k),2*(j*k-i*r),
                        2*(i*k-j*r),2*(j*k+i*r),1-2*(i*i+j*j)],dim=-1).reshape(q.shape[:-1]+(3,3))


def coherence(canonical,positions,rotations,edges,rest,weight):
    """Learn the shared canonical shape, penalize inconsistent local transforms.

    Frozen initial lengths normalize errors; they are not imposed as rest shape.
    This permits canonical geometry learning without fitting arbitrary per-time
    deformations. Both directed ends receive constraints from the full graph.
    """
    if len(edges)==0: raise ValueError('Empty graph cannot supervise geometry')
    a,b=edges.unbind(-1)
    expected=(rotations[a]@(canonical[b]-canonical[a])[...,None]).squeeze(-1)
    residual=((positions[b]-positions[a])-expected)/rest[:,None].clamp_min(1e-6)
    rigid=torch.nn.functional.huber_loss(residual,torch.zeros_like(residual),reduction='none',delta=.2).mean(-1)
    rotation=(rotations[b]-rotations[a]).square().mean((-1,-2))
    denominator=weight.sum().clamp_min(1e-8)
    return {'rigid':(weight*rigid).sum()/denominator,'rotation':(weight*rotation).sum()/denominator}


def scene_prior(scene,t,total,graph,config):
    """Only canonical points and original PIDG grid/network receive gradients."""
    count=len(graph['edges']); batch=min(count,config['edges_per_step'])
    rows=(torch.arange(batch,device=scene.centers.device)+total*batch)%count
    edges=graph['edges'][rows]
    nodes,inverse=torch.unique(edges.flatten(),sorted=True,return_inverse=True)
    local_edges=inverse.reshape(-1,2)
    canonical=scene.pools[0].get_xyz[nodes]
    state=scene.deforms[0].step(canonical,canonical.new_full((len(canonical),1),float(t)))
    rotation=quaternion_matrix(state['d_xyz'][:,:4])
    position=(rotation@canonical[...,None]).squeeze(-1)+state['d_xyz'][:,4:]+scene.centers[0]
    return coherence(canonical,position,rotation,local_edges,graph['rest'][rows],graph['weight'][rows])
