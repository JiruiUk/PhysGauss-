"""Training-only ordinal depth prior; no material/instance parameters."""
from types import SimpleNamespace
import numpy as np
import torch


def pair_bank(pseudo, mask, seed, count=20000):
    """Fixed global/local pairs, inverse-depth direction, no global RNG use."""
    yy, xx = np.nonzero(mask & np.isfinite(pseudo))
    if len(xx) < 64:
        raise ValueError('Insufficient fixed foreground depth support')
    height, width = mask.shape
    rng = np.random.default_rng(seed)
    first = rng.integers(len(xx), size=count*4)
    second = rng.integers(len(xx), size=count*4)
    a = yy[first]*width+xx[first]
    b = yy[second]*width+xx[second]
    # Add local shape order rather than only inter-body front/back order.
    x2 = xx[first]+rng.integers(-32, 33, len(first))
    y2 = yy[first]+rng.integers(-32, 33, len(first))
    valid = (x2 >= 0) & (x2 < width) & (y2 >= 0) & (y2 < height)
    ix = np.flatnonzero(valid)
    ix = ix[mask[y2[ix], x2[ix]]]
    local_a = a[ix]
    local_b = y2[ix]*width+x2[ix]
    scale = max(float(np.quantile(pseudo[mask], .9)-np.quantile(pseudo[mask], .1)), 1e-6)
    pairs, signs = [], []
    for aa, bb in [(a, b), (local_a, local_b)]:
        # Larger cached value is closer, as qualified against frozen static
        # support. Do not treat this scale-free signal as metric depth GT.
        difference = pseudo.ravel()[bb]-pseudo.ravel()[aa]
        keep = np.flatnonzero(np.abs(difference) > .02*scale)[:count]
        pairs.append(np.stack([aa[keep], bb[keep]], -1))
        signs.append(np.sign(difference[keep]).astype(np.float32))
    # Interleave both types so every scheduled batch contains both.
    n = min(map(len, pairs))
    if n < 64:
        raise ValueError('Insufficient confident local/global depth pairs')
    return np.stack([p[:n] for p in pairs], 1).reshape(-1, 2).astype(np.int64), np.stack([s[:n] for s in signs], 1).ravel()


def ordinal_loss(depth, alpha, pairs, signs, sharpness=10.):
    expected = depth.squeeze()/alpha.squeeze().clamp_min(1e-6)
    values = expected.flatten()[pairs]
    return (torch.tanh(sharpness*(values[:, 0]-values[:, 1]))-signs).abs().mean()


def geometry_only(dynamic):
    """Depth supervision changes centers/field, not appearance or opacity."""
    return SimpleNamespace(get_xyz=dynamic.get_xyz, get_scaling=dynamic.get_scaling.detach(),
        get_rotation=dynamic.get_rotation.detach(), get_opacity=dynamic.get_opacity.detach(),
        get_features=dynamic.get_features.detach(), max_sh_degree=dynamic.max_sh_degree,
        active_sh_degree=dynamic.active_sh_degree)
