"""Directed local-rigid propagation; no body labels, material law or force.

Local rotations are fitted without gradients, as a local/global ARAP step.
Gradients flow through target centers, source centroids and actual covariances.
All reference coordinates and confidences are frozen model-conditioned priors.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


def build_patches(positions, support, good_fraction, opacity, keys, trusted, *,
                  radius=.08, candidates=24, neighbors=8, common_min=3,
                  condition_min=.01, minimum_sources=3):
    x = np.asarray(positions, dtype=np.float32)
    score = np.asarray(support)
    quality = (np.asarray(good_fraction) >= .95) & (score > 1e-8)
    opacity = np.asarray(opacity).reshape(-1)
    keys = np.asarray(keys, dtype=np.int64); trusted = np.asarray(trusted, dtype=bool)
    if x.shape != (*score.shape, 3) or trusted.shape != score.shape or len(opacity) != x.shape[1]:
        raise ValueError('Expected aligned TN3 trajectories, TN scores/visibility, N opacity')
    if not all(np.isfinite(a).all() for a in [x, score, good_fraction, opacity]):
        raise ValueError('Finite frozen inputs required')
    if radius <= 0 or minimum_sources < 3 or neighbors < minimum_sources or candidates < neighbors or common_min < 2:
        raise ValueError('Invalid patch construction settings')
    times, population = score.shape
    proposals = []; encoded = []
    for t in range(times):
        target = np.flatnonzero(quality[t] & (opacity > .1))
        source = keys[trusted[t, keys]]
        if len(source) < minimum_sources or not len(target):
            continue
        d, j = cKDTree(x[t, source]).query(x[t, target], k=min(candidates, len(source)))
        source_ids = source[j]
        valid = (d <= radius) & (source_ids != target[:, None])
        code = target[:, None]*population+source_ids
        encoded.append(code[valid]); proposals.append((t, target, source_ids, d, valid, code))
    if not encoded or not any(len(a) for a in encoded):
        raise ValueError('No supported directed candidates')
    pair_codes = np.unique(np.concatenate(encoded)); compatible = np.zeros(len(pair_codes), bool)
    common_counts = np.zeros(len(pair_codes), np.int16)
    for begin in range(0, len(pair_codes), 4096):
        ids = pair_codes[begin:begin+4096]
        i, j = ids//population, ids % population
        common = quality[:, i] & trusted[:, j]
        counts = common.sum(0)
        distance = np.linalg.norm(x[:, i]-x[:, j], axis=-1)
        maximum = np.where(common, distance, 0).max(0)
        compatible[begin:begin+len(ids)] = (counts >= common_min) & (maximum <= radius)
        common_counts[begin:begin+len(ids)] = counts
    best_score = np.full(population, -1., np.float64)
    source_out = np.zeros((population, neighbors), np.int64)
    weights_out = np.zeros((population, neighbors), np.float32)
    reference_index = np.full(population, -1, np.int64)
    condition = np.zeros(population, np.float32)
    for t, target, source_ids, distance, valid, code in proposals:
        loc = np.searchsorted(pair_codes, code.clip(pair_codes[0], pair_codes[-1]))
        supported = valid & (pair_codes[loc] == code) & compatible[loc]
        for row in np.flatnonzero((supported.sum(1) >= minimum_sources) & (score[t, target] > best_score[target])):
            node = target[row]
            chosen = np.flatnonzero(supported[row])[:neighbors]
            source = source_ids[row, chosen]; d = distance[row, chosen]
            weight = np.exp(-.5*(d/radius)**2); weight /= weight.sum()
            ref = x[t, source].astype(np.float64)
            ref -= (ref*weight[:, None]).sum(0)
            eigen = np.linalg.eigvalsh((ref*weight[:, None]).T@ref)
            ratio = eigen[-2]/max(eigen[-1], 1e-12)
            if ratio < condition_min:
                continue
            best_score[node] = score[t, node]; reference_index[node] = t; condition[node] = ratio
            source_out[node, :len(source)] = source
            weights_out[node, :len(source)] = weight
            # Remaining padded slots have exactly zero weight; no filled edges.
            weights_out[node, len(source):] = 0
    target = np.flatnonzero(reference_index >= 0)
    if not len(target):
        raise ValueError('No non-collinear supported patches')
    reference = reference_index[target]; sources = source_out[target]; weight = weights_out[target]
    confidence = np.where(trusted[:, keys], 1., 0.).astype(np.float32)
    return {'targets': target, 'sources': sources, 'spatial_weights': weight,
        'reference_indices': reference, 'reference_target': x[reference, target],
        'reference_sources': x[reference[:, None], sources], 'condition': condition[target],
        'keys': keys, 'key_confidence': confidence, 'reference_key_positions': x[:, keys],
        'pair_candidate_count': np.array(len(pair_codes)), 'qualified_pair_count': np.array(compatible.sum()),
        'pair_common_count_quantiles': np.quantile(common_counts[compatible], [0, .5, 1])}


def fit_local_rotation(reference, current, weight):
    """Best proper rotation with a detached local step, finite at repeated SVDs."""
    normalized = weight/weight.sum(-1, keepdim=True).clamp_min(1e-12)
    ref_center = (reference*normalized[..., None]).sum(-2)
    current_center = (current*normalized[..., None]).sum(-2)
    with torch.no_grad():
        a = (reference-ref_center[..., None, :]).double()
        b = (current-current_center[..., None, :]).double()
        h = (a*normalized.double()[..., None]).transpose(-1, -2)@b
        u, _, vh = torch.linalg.svd(h)
        v = vh.transpose(-1, -2)
        correction = torch.ones(*h.shape[:-2], 3, device=h.device, dtype=h.dtype)
        correction[..., -1] = torch.linalg.det(v@u.transpose(-1, -2)).sign()
        rotation = (v@torch.diag_embed(correction)@u.transpose(-1, -2)).to(reference.dtype)
    return rotation, ref_center, current_center


def robust_length(vector, epsilon=1e-4):
    return (vector.square().sum(-1)+epsilon**2).sqrt()-epsilon


def propagate(positions, covariances, graph, frame_index, target_rows, *, uniform=False,
              confidence_floor=.05, position_scale=.08, depth_precision=.01):
    ids = graph['targets'][target_rows]; sources = graph['sources'][target_rows]
    base_weight = graph['spatial_weights'][target_rows]
    confidence = positions.new_ones(len(graph['keys'])) if uniform else graph['key_confidence'][frame_index]
    lookup = graph['key_lookup'][sources]
    source_conf = confidence[lookup]
    weight = base_weight*(confidence_floor+(1-confidence_floor)*source_conf)
    reference = graph['reference_sources'][target_rows]
    rotation, ref_center, current_center = fit_local_rotation(reference, positions[sources], weight)
    delta = graph['reference_target'][target_rows]-ref_center
    predicted = current_center+(rotation@delta[..., None]).squeeze(-1)
    # Frozen weights: current opacity or appearance cannot disable this prior.
    reliability = weight.sum(-1)
    center_loss = (reliability*robust_length((positions[ids]-predicted)/position_scale)).mean()
    covariance_ref = graph['reference_covariance'][target_rows]
    predicted_covariance = rotation@covariance_ref@rotation.transpose(-1, -2)
    size = covariance_ref.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-8)
    error = ((covariances[ids]-predicted_covariance)/size[:, None, None]).flatten(1)
    covariance_loss = (reliability*robust_length(error)).mean()
    difference = positions[graph['keys']]-graph['reference_key_positions'][frame_index]
    camera_difference = difference@graph['camera_rotation'][frame_index].T
    scale = positions.new_tensor([1., 1., depth_precision**.5])
    tether = (confidence*robust_length(camera_difference*scale/position_scale)).sum()/max(1, len(confidence))
    return {'center': center_loss, 'covariance': covariance_loss, 'tether': tether}, {
        'patches': len(ids), 'mean_source_confidence': float((base_weight*source_conf).sum(-1).mean()),
        'mean_frozen_patch_reliability': float(reliability.mean()),
        'currently_trusted_keys': int((confidence > 0).sum())}


def extend(original, enabled, callback):
    """Flag off returns the exact existing scalar object without querying a prior."""
    return original+callback() if enabled else original
