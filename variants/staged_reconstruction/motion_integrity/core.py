"""Frozen trajectory-proximity neighborhoods and rotation-invariant strain prior.

This is a conditional reconstruction prior, not material correspondence truth.
There are no instance labels, material parameters or force unknowns.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


def trajectory_graph(positions, k=12, radius=.08, tolerance=.2):
    x = np.asarray(positions, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 3 or not np.isfinite(x).all():
        raise ValueError('Expected finite time x population x 3 positions')
    times, n, _ = x.shape
    if times < 2 or n < 2 or k < 1 or radius <= 0 or not 0 <= tolerance < 1:
        raise ValueError('Invalid graph inputs')
    candidates = []
    for frame in x:
        _, neighbors = cKDTree(frame).query(frame, k=min(k+1, n), workers=1)
        source = np.repeat(np.arange(n), neighbors.shape[1])
        target = neighbors.reshape(-1)
        edges = np.sort(np.stack([source, target], -1), -1)
        candidates.append(edges[edges[:, 0] != edges[:, 1]])
    edges = np.unique(np.concatenate(candidates), axis=0)
    maximum, median = [], []
    for start in range(0, len(edges), 8192):
        pairs = edges[start:start+8192]
        distances = np.linalg.norm(x[:, pairs[:, 0]]-x[:, pairs[:, 1]], axis=-1)
        maximum.append(distances.max(0))
        median.append(np.median(distances, axis=0))
    maximum = np.concatenate(maximum)
    rest = np.concatenate(median)
    good = (maximum <= radius) & (rest > 1e-6)
    edges, maximum, rest = edges[good], maximum[good], rest[good]
    # Mutual k nearest under maximum trajectory separation (MoSca-inspired).
    # All candidate times affect the curve distance; contact-only edges fail.
    selected = np.zeros(len(edges), dtype=np.int8)
    directed_nodes = np.concatenate([edges[:, 0], edges[:, 1]])
    directed_edges = np.tile(np.arange(len(edges)), 2)
    order = np.lexsort((directed_edges, maximum[directed_edges], directed_nodes))
    nodes = directed_nodes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(nodes))+1] if len(nodes) else []
    for start, end in zip(starts, list(starts[1:])+[len(nodes)]):
        selected[directed_edges[order[start:min(end, start+k)]]] += 1
    good = selected == 2
    edges, maximum, rest = edges[good], maximum[good], rest[good]
    weight = np.exp(-.5*(maximum/radius)**2).astype(np.float32)
    return {'edges': edges.astype(np.int64), 'rest': rest.astype(np.float32),
            'weight': weight, 'curve_max': maximum, 'tolerance': float(tolerance),
            'population': n, 'times': times}


def strain_loss(positions, edges, rest, weight, tolerance=.2):
    """Fixed-denominator robust log-strain; no trainable confidence/opacity escape.

    Symmetric log tolerance permits moderate compression/extension. The rest
    lengths are frozen medians of parent trajectories, not canonical geometry GT.
    """
    if len(edges) == 0:
        raise ValueError('No qualified edges; an empty graph is not supervision')
    distance = (positions[edges[:, 1]]-positions[edges[:, 0]]).norm(dim=-1)
    log_strain = torch.log(distance.clamp_min(1e-7)/rest.clamp_min(1e-7))
    excess = (log_strain.abs()-np.log1p(tolerance)).clamp_min(0)
    penalty = torch.nn.functional.huber_loss(excess, torch.zeros_like(excess), reduction='none', delta=.2)
    return (weight*penalty).sum()/weight.sum().clamp_min(1e-8)
