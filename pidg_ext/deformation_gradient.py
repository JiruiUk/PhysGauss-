"""Canonical-neighbour weighted affine fits; no neural coordinate derivatives."""
import torch


def canonical_neighbors(X, num_neighbors=16):
    if X.ndim != 2 or X.shape[1] != 3 or len(X) < 2:
        raise ValueError('Expected at least two canonical 3D positions')
    k = min(num_neighbors, len(X)-1)
    if k < 1:
        raise ValueError('num_neighbors must be positive')
    try:
        from scipy.spatial import cKDTree
        _, indices = cKDTree(X.detach().cpu().double().numpy()).query(
            X.detach().cpu().double().numpy(), k=k+1)
        indices = torch.as_tensor(indices, device=X.device, dtype=torch.long)
        # Remove self explicitly: duplicate points need not return self first.
        self_ids = torch.arange(len(X), device=X.device)[:,None]
        keep = indices != self_ids
        order = torch.argsort((~keep).long(), dim=1, stable=True)
        return indices.gather(1,order)[:,:k]
    except ImportError:
        rows = []
        for start in range(0,len(X),256):
            distances = torch.cdist(X[start:start+256].detach(),X.detach())
            local = torch.arange(len(distances), device=X.device)
            distances[local,local+start] = float('inf')
            rows.append(distances.topk(k,largest=False).indices)
        return torch.cat(rows)


def estimate_F(X, x, neighbors, query_indices=None, regularization=1e-4,
               min_valid_neighbors=6, max_condition_number=1e6, min_determinant=1e-6):
    if X.shape != x.shape or X.ndim != 2 or X.shape[1] != 3:
        raise ValueError('X and x must have matching [N,3] shapes')
    if not torch.isfinite(X).all() or not torch.isfinite(x).all():
        raise ValueError('Nonfinite positions')
    if neighbors.ndim != 2 or neighbors.shape[0] != len(X) or neighbors.dtype != torch.long:
        raise ValueError('neighbors must be [N,K] integer indices')
    if neighbors.numel()==0 or neighbors.min()<0 or neighbors.max()>=len(X):
        raise ValueError('Invalid neighbor indices')
    if regularization <= 0:
        raise ValueError('regularization must be positive')
    ids = torch.arange(len(X),device=X.device) if query_indices is None else query_indices
    graph = neighbors[ids]
    dX = X[graph]-X[ids,None]
    dx = x[graph]-x[ids,None]
    distance2 = dX.square().sum(-1)
    usable = (distance2 > 1e-12) & (graph != ids[:,None])
    bandwidth = distance2.detach().mean(-1,keepdim=True).clamp_min(1e-12)
    weights = torch.exp(-distance2/bandwidth)*usable
    A = torch.einsum('nk,nki,nkj->nij',weights,dX,dX)
    B = torch.einsum('nk,nki,nkj->nij',weights,dx,dX)
    eigenvalues = torch.linalg.eigvalsh(A.detach())
    condition = eigenvalues[:,-1]/eigenvalues[:,0].clamp_min(torch.finfo(X.dtype).tiny)
    condition = torch.where(eigenvalues[:,0]>0,condition,torch.full_like(condition,float('inf')))
    eye = torch.eye(3,device=X.device,dtype=X.dtype).expand(len(ids),3,3)
    ridge = regularization*A.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-12)
    # An identity prior gives exact identity at rest and finite singular-neighbour solves.
    F = torch.linalg.solve(A+ridge[:,None,None]*eye,
                           (B+ridge[:,None,None]*eye).transpose(-1,-2)).transpose(-1,-2)
    J = torch.linalg.det(F)
    valid = ((usable.sum(-1)>=min_valid_neighbors) & (condition<=max_condition_number)
             & (eigenvalues[:,0]>1e-12) & (J>min_determinant) & torch.isfinite(F).all(dim=(-1,-2)))
    confidence = (valid.to(X.dtype)/(1+condition/max_condition_number)).detach()
    return {'F':F,'J':J,'valid':valid.detach(),'confidence':confidence,
            'condition_number':condition,'query_indices':ids}


class CanonicalGraph:
    """Cache canonical topology, rebuilding after densification or a scheduled refresh."""
    def __init__(self, num_neighbors=16, refresh_interval=1000):
        self.num_neighbors=num_neighbors
        self.refresh_interval=refresh_interval
        self.neighbors=None
        self.last_iteration=None

    def get(self,X,iteration):
        if (self.neighbors is None or len(self.neighbors)!=len(X) or
                iteration-self.last_iteration>=self.refresh_interval):
            self.neighbors=canonical_neighbors(X,self.num_neighbors)
            self.last_iteration=iteration
        return self.neighbors
