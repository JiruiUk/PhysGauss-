"""Pure differentiable objectives; no material law, force or segmentation."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def static_position(now, reference):
    if now.shape != reference.shape or now.shape[-1] != 3:
        raise ValueError('Expected matching Nx3 positions')
    return F.smooth_l1_loss(now, reference, beta=.01)


def static_anchor(positions, targets, tolerance=.02):
    distance = torch.linalg.vector_norm(positions-targets.detach(), dim=-1)
    return F.smooth_l1_loss(F.relu(distance-tolerance), torch.zeros_like(distance), beta=.02)


def static_covariance(now, reference):
    denominator = reference.detach().square().sum((-1,-2)).sqrt().clamp_min(1e-6)
    value = (now-reference)/denominator[...,None,None]
    return F.smooth_l1_loss(value, torch.zeros_like(value), beta=.1)


def local_shape(now_a, now_b, reference_length, lower=.5, upper=2.):
    if not 0 < lower < 1 < upper:
        raise ValueError('Invalid allowed local stretch')
    length = torch.linalg.vector_norm(now_b-now_a,dim=-1).clamp_min(1e-6)
    ratio = torch.log(length/reference_length.detach().clamp_min(1e-6))
    error = F.relu(math.log(lower)-ratio)+F.relu(ratio-math.log(upper))
    return F.smooth_l1_loss(error, torch.zeros_like(error), beta=.1)


def project(positions, view, intrinsic):
    camera = positions@view[:3,:3].T+view[:3,3]
    pixel = camera@intrinsic.T
    return pixel[:,:2]/camera[:,2:3].clamp_min(.05), camera[:,2]


def reprojection(positions, view, intrinsic, target):
    uv, depth = project(positions,view,intrinsic)
    # Camera calibration/targets are constants; moving behind the camera is not
    # a way to remove supervision. Target visibility was frozen before training.
    residual = (uv-target.detach())/uv.new_tensor([400.,225.])
    loss = F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=1/400)
    return loss+F.relu(.2-depth).mean(), (uv-target.detach()).norm(dim=-1).mean()


def fixed_edges(positions, ids, local_labels, k=6, radius=.08):
    """Known same-clip/same-region only; unknown labels never acquire edges."""
    from scipy.spatial import cKDTree
    positions=np.asarray(positions); ids=np.asarray(ids); labels=np.asarray(local_labels)
    edges=[]
    for group in np.unique(labels[labels>=0]):
        use=ids[labels==group]
        if len(use)<2:continue
        distances,neighbors=cKDTree(positions[use]).query(positions[use],k=min(k+1,len(use)))
        for row in range(len(use)):
            for d,j in zip(np.atleast_1d(distances[row])[1:],np.atleast_1d(neighbors[row])[1:]):
                if 1e-3<d<radius and use[row]<use[j]:edges.append((use[row],use[j],d))
    return np.asarray(edges,dtype=np.float64).reshape(-1,3)
