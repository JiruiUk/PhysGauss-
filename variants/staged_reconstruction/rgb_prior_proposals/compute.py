"""One saved candidate, from raw prior plus RGB only; no evaluator imports."""
import numpy as np
import torch
from .core import ordinal_bounds, propose


def compute_view(depth, anchors, raw, features, texture, views, k, index, sources, near, far, protocol):
    yy, xx = np.nonzero(depth > 0); uv = np.c_[xx, yy]; old = depth[yy, xx]
    assert np.array_equal(anchors['uv'], uv) and np.array_equal(anchors['depth'], old)
    tight = anchors['tight']
    lo, hi, known, lo_blocks, hi_blocks = ordinal_bounds(uv, raw, anchors['uv'][tight],
        anchors['lower'][tight], anchors['upper'][tight], near, far, protocol)
    output = old.copy(); diagnostics = {}
    for start in range(0, len(old), 256):
        section = slice(start, start+256)
        xyz = torch.as_tensor(uv[section], dtype=torch.float32, device=features.device)
        z = torch.as_tensor(old[section], device=features.device)
        disabled, off = propose(None, z)
        assert disabled is z and not off['accepted'].any()
        result, values = propose(xyz, z, torch.as_tensor(lo[section], dtype=z.dtype, device=z.device),
            torch.as_tensor(hi[section], dtype=z.dtype, device=z.device), torch.as_tensor(known[section], device=z.device),
            features, texture, views, k, index, sources, near, far, True)
        output[section] = result.cpu().numpy()
        for name, value in values.items(): diagnostics.setdefault(name, []).append(value.cpu().numpy())
    saved = {name: np.concatenate(value, axis=1 if name.startswith('held_') else 0) for name, value in diagnostics.items()}
    assert np.array_equal(output[~saved['accepted']], old[~saved['accepted']])
    assert np.array_equal(output != old, saved['accepted'])
    arrays = {'uv': uv, 'original': old, 'updated': output, 'lower': lo, 'upper': hi, 'known': known,
              'lower_anchor_blocks': lo_blocks, 'upper_anchor_blocks': hi_blocks, **saved}
    counts = {'candidates': len(old), 'bounded': int(known.sum()), 'eligible': int(saved['eligible'].sum()),
              'finite_proposals': int(np.isfinite(saved['proposed']).sum()), 'accepted': int(saved['accepted'].sum())}
    return arrays, counts
