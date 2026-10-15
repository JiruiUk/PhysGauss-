"""Canonical shared-normal traction continuity for known material neighbours."""
import torch


def interface_pairs(X,material_ids,neighbors,max_distance=float('inf')):
    if neighbors.ndim!=2 or len(neighbors)!=len(X):raise ValueError('Invalid neighbour graph')
    source=torch.arange(len(X),device=X.device)[:,None].expand_as(neighbors)
    known=(neighbors>=0)&(neighbors<len(X))
    target=neighbors.clamp(0,len(X)-1)
    distance2=(X[target]-X[:,None]).square().sum(-1)
    keep=(known & (source!=target) & (material_ids[:,None]!=material_ids[target])
          & (distance2>1e-12) & (distance2<=max_distance**2))
    pairs=torch.stack((source[keep],target[keep]),dim=-1)
    return torch.unique(pairs.sort(dim=-1).values,dim=0)


def traction_loss(sigma,X,pairs,confidence=None):
    if len(pairs)==0:return sigma.sum()*0
    left,right=pairs.unbind(-1)
    delta=X[right]-X[left]
    normal=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    difference=((sigma[left]-sigma[right])@normal[...,None]).squeeze(-1)
    error=difference.square().sum(-1)
    if confidence is not None:
        error=error*torch.minimum(confidence[left],confidence[right]).detach()
    return error.mean()
