"""SC-GS-inspired interpolation with the complete original PIDG center mapping."""
import numpy as np
import torch
from scipy.spatial import cKDTree


def controls_from_canonical(xyz, count=256):
    x = np.asarray(xyz, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or not 4 <= count <= len(x):
        raise ValueError('Invalid canonical control population')
    chosen, nearest = [], np.full(len(x), np.inf)
    index = int(np.argmax(np.sum((x-x.mean(0))**2, axis=-1)))
    for _ in range(count):
        chosen.append(index)
        nearest = np.minimum(nearest, np.sum((x-x[index])**2, axis=-1))
        nearest[chosen] = -1
        index = int(np.argmax(nearest))
    nodes = x[chosen]
    distances, _ = cKDTree(nodes).query(nodes, k=4, workers=1)
    radius = np.maximum(distances[:, -1]*.5, 1e-4)
    return np.array(chosen), nodes, radius.astype(np.float32)


def matrix(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    r, i, j, k = q.unbind(-1)
    return torch.stack([1-2*(j*j+k*k), 2*(i*j-k*r), 2*(i*k+j*r),
                        2*(i*j+k*r), 1-2*(i*i+k*k), 2*(j*k-i*r),
                        2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i*i+j*j)], -1).reshape(*q.shape[:-1], 3, 3)


def weights(query, controls, radius, k=4):
    distances, ids = torch.cdist(query, controls).square().topk(min(k, len(controls)), largest=False)
    logits = -distances/(2*radius[ids].square().clamp_min(1e-8))
    return ids, logits.softmax(-1)


def aligned_mean(quaternions, w):
    q = torch.nn.functional.normalize(quaternions, dim=-1)
    sign = torch.where((q*q[:, :1]).sum(-1, keepdim=True) < 0, -1., 1.)
    return torch.nn.functional.normalize((q*sign*w[..., None]).sum(1), dim=-1)


def interpolate(query, node_state, ids, w):
    q = node_state['d_xyz'][ids, :4]
    translation = node_state['d_xyz'][ids, 4:]
    transformed = (matrix(q)@query[:, None, :, None]).squeeze(-1)+translation
    center = (w[..., None]*transformed).sum(1)
    rotation = aligned_mean(q, w)
    # API must encode the same full center for rendering AND physics queries.
    offset = center-(matrix(rotation)@query[..., None]).squeeze(-1)
    return {'d_xyz': torch.cat([rotation, offset], -1),
            'd_rotation': (w[..., None]*node_state['d_rotation'][ids]).sum(1),
            'd_scaling': (w[..., None]*node_state['d_scaling'][ids]).sum(1),
            'reg_spatial': None, 'reg_temporal': None}


class SparseDeform:
    """Same field parameters, evaluated only at controls when mix=1.

    Canonical controls/radii are frozen and shared at every time. KNN/RBF weights
    depend on the query coordinate, not camera, opacity, time or instance IDs.
    This is a representation prior, not the full SC-GS training method.
    """
    def __init__(self, original, controls, radius, mix=0.):
        self.original, self.controls, self.radius = original, controls, radius
        self.mix = float(mix)

    @property
    def reg_spatial_able(self):
        return self.original.reg_spatial_able

    @reg_spatial_able.setter
    def reg_spatial_able(self, value):
        self.original.reg_spatial_able = value

    @property
    def reg_temporal_able(self):
        return self.original.reg_temporal_able

    @reg_temporal_able.setter
    def reg_temporal_able(self, value):
        self.original.reg_temporal_able = value

    def step(self, xyz, t, fixed_attention=False):
        if self.mix == 0:
            return self.original.step(xyz, t, fixed_attention=fixed_attention)
        if not 0 <= self.mix <= 1:
            raise ValueError('Sparse interpolation mix must lie in [0,1]')
        if not torch.equal(t, t[:1].expand_as(t)):
            raise ValueError('Queries must share one time')
        node = self.original.step(self.controls, t[:1].expand(len(self.controls), 1), fixed_attention=False)
        ids, w = weights(xyz, self.controls, self.radius)
        sparse = interpolate(xyz, node, ids, w)
        if self.mix == 1:
            return sparse
        dense = self.original.step(xyz, t, fixed_attention=False)
        q0 = torch.nn.functional.normalize(dense['d_xyz'][:, :4], dim=-1)
        q1 = sparse['d_xyz'][:, :4]
        q1 = torch.where((q0*q1).sum(-1, keepdim=True) < 0, -q1, q1)
        q = torch.nn.functional.normalize((1-self.mix)*q0+self.mix*q1, dim=-1)
        old_center = (matrix(q0)@xyz[..., None]).squeeze(-1)+dense['d_xyz'][:, 4:]
        new_center = (matrix(q1)@xyz[..., None]).squeeze(-1)+sparse['d_xyz'][:, 4:]
        center = (1-self.mix)*old_center+self.mix*new_center
        return {'d_xyz': torch.cat([q, center-(matrix(q)@xyz[..., None]).squeeze(-1)], -1),
                'd_rotation': (1-self.mix)*dense['d_rotation']+self.mix*sparse['d_rotation'],
                'd_scaling': (1-self.mix)*dense['d_scaling']+self.mix*sparse['d_scaling'],
                'reg_spatial': None, 'reg_temporal': None}
