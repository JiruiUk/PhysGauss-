"""Pure subset operations; no learned segmentation, material law or new parameters."""
import torch


def validate_rows(ids, count):
    if ids.ndim != 1 or ids.dtype != torch.long:
        raise ValueError('Static row IDs must be a one-dimensional int64 tensor')
    if not len(ids) or bool((ids < 0).any()) or bool((ids >= count).any()):
        raise ValueError('Empty or out-of-range static pool')
    if len(torch.unique(ids)) != len(ids):
        raise ValueError('Duplicate static row IDs')


def replace_static(warp, ids, reference):
    """Replace only known-static rows; preserve dynamic tensors and their gradients."""
    keys = ('d_xyz', 'd_rotation', 'd_scaling')
    count = warp['d_xyz'].shape[0]
    validate_rows(ids, count)
    result = dict(warp)
    for key, width in zip(keys, (7, 4, 3)):
        value, fixed = warp[key], reference[key]
        if value.shape != (count, width) or fixed.shape != (len(ids), width):
            raise ValueError('Warp/reference shape mismatch: ' + key)
        if not bool(torch.isfinite(fixed).all()):
            raise ValueError('Nonfinite static reference: ' + key)
        result[key] = value.index_copy(0, ids, fixed.detach().to(value))
    return result


def dynamic_rows(ids, count):
    validate_rows(ids, count)
    keep = torch.ones(count, dtype=torch.bool, device=ids.device)
    keep[ids] = False
    return torch.nonzero(keep, as_tuple=False).flatten()


def select_physics_permutation(eligible, count, sample_size, device):
    """Return original-row indices; disabled callers use the original code path."""
    if eligible is None:
        return torch.randperm(count, device=device)[:min(count, sample_size)], count
    if eligible.ndim != 1 or eligible.dtype != torch.long:
        raise ValueError('Invalid dynamic sampling rows')
    n = len(eligible)
    return eligible[torch.randperm(n, device=device)[:min(n, sample_size)]], n
