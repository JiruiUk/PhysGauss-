import torch
import torch.nn.functional as F
from variants.flow_comparison.geometry import sample


def anchor_objective(groups, masks, reference, floor=.8):
    if groups.shape != masks.shape or groups.ndim != 3 or groups.shape[0] != 4:
        raise ValueError('Expected four aligned group contribution images and masks')
    mass = groups.sum((1, 2))
    inside = (groups * masks).sum((1, 2))
    valid = (masks.sum((1, 2)) > 0) & (reference > 0)
    leakage = ((mass - inside).clamp_min(0) / mass.detach().clamp_min(1))[valid]
    coverage = inside / masks.sum((1, 2)).clamp_min(1)
    missing = (F.relu(reference * floor - coverage) / reference.clamp_min(1e-6)).square()[valid]
    zero = groups.sum() * 0
    return {'leakage': leakage.mean() if valid.any() else zero,
            'coverage_floor': missing.mean() if valid.any() else zero}


def temporal_objective(source, target, forward, valid):
    if source.shape != target.shape or source.ndim != 3 or source.shape[0] != 4:
        raise ValueError('Expected matching four-channel group images')
    h, w = forward.shape[:2]
    if tuple(source.shape[1:]) != (h, w) or tuple(valid.shape) != (h, w):
        raise ValueError('Group images and flow use different pixel coordinates')
    y, x = torch.meshgrid(torch.arange(h, device=forward.device, dtype=forward.dtype),
                          torch.arange(w, device=forward.device, dtype=forward.dtype), indexing='ij')
    coords = torch.stack([x, y], -1) + forward
    pulled = sample(target.permute(1, 2, 0), coords).permute(2, 0, 1)
    mask = valid[None].to(source)
    denominator = ((source.detach() + pulled.detach()) * mask).sum().clamp_min(1)
    return ((source - pulled).abs() * mask).sum() / denominator
