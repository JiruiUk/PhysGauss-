"""Common Lagrangian Cauchy discretization in current world coordinates.

Both comparison arms use this operator. Predicted PIDG stress and its original
deviatoric rate correction are retained; no constitutive law or force is added.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


def fit_gradient(x, values, anchors, neighbors, ridge=1e-5):
    dx = x[neighbors]-x[anchors, None]
    dy = values[neighbors]-values[anchors, None]
    # Relative distance weights, detached to avoid optimizing the sampling rule.
    d2 = dx.detach().square().sum(-1)
    weight = 1/(d2+d2.mean(-1, keepdim=True).clamp_min(1e-12)*.1)
    matrix = torch.einsum('nk,nki,nkj->nij', weight, dx, dx)
    eigen = torch.linalg.eigvalsh(matrix.detach())
    valid = eigen[:, 0] > 1e-4*eigen[:, -1].clamp_min(1e-12)
    eye = torch.eye(3, device=x.device, dtype=x.dtype)
    regularized = matrix+ridge*matrix.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None].clamp_min(1e-12)*eye/3
    rhs = torch.einsum('nk,nki,nkc->nic', weight, dx, dy)
    return torch.linalg.solve(regularized, rhs).transpose(-1, -2), valid


def stress_tensor(params, grad_v, coefficient):
    a, b, c, d, e, f = params.unbind(-1)
    stress = torch.stack([a, d, e, d, b, f, e, f, c], -1).reshape(-1, 3, 3)
    rate = .5*(grad_v+grad_v.transpose(-1, -2))
    eye = torch.eye(3, device=rate.device, dtype=rate.dtype)
    return stress+coefficient*(rate-rate.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]*eye/3)


class Graph:
    def __init__(self):
        self.neighbors = None
        self.last = -10000

    def sample(self, canonical, iteration, count=64, k=12):
        n = len(canonical)
        if self.neighbors is None or len(self.neighbors) != n or iteration-self.last >= 1000:
            points = canonical.detach().cpu().numpy()
            tree = cKDTree(points)
            _, ids = tree.query(points, k=min(k+1, n), workers=1)
            self.neighbors = torch.as_tensor(np.asarray(ids[:, 1:]), device=canonical.device)
            self.last = iteration
        anchors = (torch.arange(min(count, n), device=canonical.device)*n//min(count, n)+iteration//100) % n
        first = torch.unique(torch.cat([anchors, self.neighbors[anchors].flatten()]))
        all_ids = torch.unique(torch.cat([first, self.neighbors[first].flatten()]))
        # Global -> all-node and first-ring indices, without dense N x N storage.
        return all_ids, first, anchors, self.neighbors[first], self.neighbors[anchors]


def residual(x, v, stress, acceleration, coefficient, all_ids, first, anchors, neighbor_first, neighbor_anchor):
    first_local = torch.searchsorted(all_ids, first)
    grad_v, good_first = fit_gradient(x, v, first_local, torch.searchsorted(all_ids, neighbor_first))
    sigma = stress_tensor(stress[first_local], grad_v, coefficient)
    anchor_first = torch.searchsorted(first, anchors)
    neighbor_first_indices = torch.searchsorted(first, neighbor_anchor)
    grad_sigma, good_anchor = fit_gradient(x[first_local], sigma.flatten(1), anchor_first, neighbor_first_indices)
    divergence = grad_sigma.reshape(-1, 3, 3, 3).diagonal(dim1=-2, dim2=-1).sum(-1)
    valid = good_anchor & good_first[anchor_first] & good_first[neighbor_first_indices].all(-1)
    r = acceleration[torch.searchsorted(all_ids, anchors)]-divergence
    # Always report the valid fraction. Degenerate neighborhoods are not evidence
    # of satisfying the equation and cannot produce a falsely good diagnostic.
    return r, valid
