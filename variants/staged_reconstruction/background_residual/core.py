"""Change a conditional visible-static reference without inventing geometry."""
import torch


def conditional_static_mask(gt):
    r, g, b = gt
    return (b > r*1.3+.02) & (b > g*1.05)


def neutralized_target(gt, static_render, mask, enabled=False):
    """Preserve foreground/empty pixels and detach the frozen static reference.

    This is an image-space training reference only, not a surface/depth label.
    A dynamic contribution in front of a blue visible surface is still penalized
    if it alters the composite relative to the unchanged frozen static render.
    """
    if not enabled:
        return gt
    if gt.ndim != 3 or gt.shape[0] != 3 or static_render.shape != gt.shape:
        raise ValueError('Expected equal CHW RGB tensors')
    if mask.shape != gt.shape[1:] or mask.dtype != torch.bool:
        raise ValueError('Expected HW Boolean visible-static mask')
    if not torch.isfinite(gt).all() or not torch.isfinite(static_render).all():
        raise ValueError('Training references must be finite')
    return torch.where(mask[None], static_render.detach(), gt)


def reconstruction(package, gt, rgba, scene, cam, *, enabled=False, static_render=None):
    # Import lazily, so unit tests of the reference transform remain independent
    # of the CUDA renderer. Disabled calls take the original expression verbatim.
    from ..union.train import reconstruction as original
    if not enabled:
        return original(package, gt, rgba, scene, cam)
    from ..stage2.train import objective
    from ..union.core import dynamic_mask, union_coverage
    static = conditional_static_mask(gt)
    target = neutralized_target(gt, static_render, static, enabled=True)
    color = objective(package['render'], target)
    mask = torch.as_tensor(dynamic_mask(rgba), device=gt.device)
    foreground = (package['render'][:, mask]-gt[:, mask]).abs().mean() if mask.any() else color*0
    static_rgb = (package['render'][:, static]-target[:, static]).abs().mean() if static.any() else color*0
    coverage = union_coverage(package['means3D_deform'][len(scene.static.get_xyz):], cam, mask)
    return color+.5*foreground+.5*static_rgb+2.*coverage
