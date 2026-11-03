"""A learned ordering nominates depths; original RGB sources alone accept them."""
import numpy as np
import torch
from ..adaptive_static.geometry import source_cost


def ordinal_bounds(uv, raw, anchor_uv, anchor_lower, anchor_upper, near, far, protocol):
    """No semantics/GT; absent or inconsistent support remains unknown."""
    n = len(uv)
    lower = np.full(n, near, np.float64); upper = np.full(n, far, np.float64)
    lo_blocks = np.zeros(n, np.int32); hi_blocks = lo_blocks.copy()
    known = np.zeros(n, bool)
    if not len(anchor_uv) or not np.isfinite(raw).all():
        return lower, upper, known, lo_blocks, hi_blocks
    span = float(np.diff(np.quantile(raw, [.05, .95]))[0])
    if span <= 1e-8:
        return lower, upper, known, lo_blocks, hi_blocks
    order = np.lexsort((anchor_uv[:, 0], anchor_uv[:, 1]))
    if len(order) > 256: order = order[np.linspace(0, len(order)-1, 256).astype(int)]
    anchors = np.asarray(anchor_uv)[order]
    lo, hi = np.asarray(anchor_lower)[order], np.asarray(anchor_upper)[order]
    valid = np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi >= lo)
    anchors, lo, hi = anchors[valid], lo[valid], hi[valid]
    if not len(anchors): return lower, upper, known, lo_blocks, hi_blocks
    block = anchors // protocol['anchor_spatial_block_pixels']
    _, block_ids = np.unique(block, axis=0, return_inverse=True)
    ar = raw[anchors[:, 1], anchors[:, 0]]
    margin = span * protocol['rank_margin_fraction_full_image_q95_q05']
    target = raw[uv[:, 1], uv[:, 0]]
    minimum = protocol['minimum_distinct_anchor_blocks_per_bound']
    for i, value in enumerate(target):
        farther = value <= ar-margin
        closer = value >= ar+margin
        lo_blocks[i] = len(np.unique(block_ids[farther]))
        hi_blocks[i] = len(np.unique(block_ids[closer]))
        has_lo, has_hi = lo_blocks[i] >= minimum, hi_blocks[i] >= minimum
        if has_lo: lower[i] = max(near, float(np.quantile(lo[farther], .8)))
        if has_hi: upper[i] = min(far, float(np.quantile(hi[closer], .2)))
        known[i] = (has_lo or has_hi) and np.isfinite(lower[i]+upper[i]) and lower[i] < upper[i]
    return lower, upper, known, lo_blocks, hi_blocks


def eligible_old_depth(depth, lower, upper, known):
    return known & ((depth < lower-1e-6) | (depth > upper+1e-6))


def accept_rgb(old, new, fit_old, fit_new, held_old, held_new, valid, enabled=False):
    if not enabled: return old, torch.zeros_like(valid)
    accepted = valid & torch.isfinite(old) & (old > 0) & torch.isfinite(new) & (new > 0)
    accepted &= torch.isfinite(fit_old) & torch.isfinite(fit_new) & (fit_new <= fit_old+1e-6)
    accepted &= torch.isfinite(held_old).all(0) & torch.isfinite(held_new).all(0)
    accepted &= (held_new <= held_old+1e-6).all(0) & (held_new < held_old-1e-6).any(0)
    accepted &= new != old
    return torch.where(accepted, new, old), accepted


@torch.no_grad()
def propose(uv, old, lower=None, upper=None, known=None, features=None, texture=None,
            views=None, k=None, index=None, sources=None, near=None, far=None, enabled=False):
    if not enabled: return old, {'accepted': torch.zeros_like(old, dtype=torch.bool)}
    assert len(sources) == 6
    assert torch.isfinite(old).all() and (old > 0).all()
    n = len(old); accepted = torch.zeros(n, device=old.device, dtype=torch.bool)
    eligible = known & (lower < upper) & ((old < lower-1e-6) | (old > upper+1e-6))
    empty = torch.full_like(old, float('nan'))
    diagnostics = {'accepted': accepted, 'eligible': eligible, 'proposed': empty.clone(),
        'fit_old': empty.clone(), 'fit_new': empty.clone(),
        'held_old': empty.repeat(2, 1), 'held_new': empty.repeat(2, 1)}
    ids = torch.nonzero(eligible).flatten()
    if not len(ids): return old.clone(), diagnostics
    xy = uv[ids]; z = old[ids]
    lo, hi = lower[ids], upper[ids]
    reference = features[index, :, xy[:, 1].long(), xy[:, 0].long()]
    rays = torch.cat([xy, torch.ones_like(xy[:, :1])], 1) @ torch.linalg.inv(k).T
    inverse = torch.linalg.inv(views[index])
    def points(depth):
        return (rays[None]*depth[..., None]) @ inverse[:3, :3].T + inverse[:3, 3]
    def fit(depth):
        cost = torch.stack([source_cost(points(depth), reference, features[j], texture[j], views[j], k) for j in sources[:4]])
        return cost.sort(0).values[:3].mean(0)
    def held(depth):
        return torch.stack([source_cost(points(depth[None]), reference, features[j], texture[j], views[j], k)[0] for j in sources[4:]])
    grid = 1./torch.linspace(1/near, 1/far, 192, device=old.device, dtype=old.dtype)
    depths = grid[:, None].expand(-1, len(ids))
    inside = (depths >= lo[None]) & (depths <= hi[None])
    costs = fit(depths).masked_fill(~inside, float('inf'))
    best, which = costs.min(0)
    proposed = grid[which]
    fit_old = fit(z[None])[0]
    held_old, held_new = held(z), held(proposed)
    valid = inside.any(0) & torch.isfinite(best)
    updated, mask = accept_rgb(z, proposed, fit_old, best, held_old, held_new, valid, True)
    result = old.clone(); result[ids] = updated
    diagnostics['accepted'][ids] = mask
    diagnostics['proposed'][ids] = torch.where(valid, proposed, torch.full_like(proposed, float('nan')))
    diagnostics['fit_old'][ids], diagnostics['fit_new'][ids] = fit_old, best
    diagnostics['held_old'][:, ids], diagnostics['held_new'][:, ids] = held_old, held_new
    return result, diagnostics
