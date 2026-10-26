"""Small differentiable objectives, independent of the PIDG runtime."""
import torch
import torch.nn.functional as F


def mask_loss(alpha, target):
    if alpha.shape != target.shape:
        raise ValueError('Alpha and foreground target must have identical shapes')
    if not torch.isfinite(alpha).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite alpha or mask')
    if (target < 0).any() or (target > 1).any():
        raise ValueError('Foreground target is outside [0,1]')
    return (alpha-target.detach()).abs().mean()


def interior(mask, radius=3):
    # Zero padding is background, including at the image border.
    padded = F.pad(mask[None], (radius,)*4, value=0.)
    return -F.max_pool2d(-padded, 2*radius+1, stride=1)[0] > .95


def coverage_metrics(alpha, target, rgb=None, gt=None):
    fg, predicted = target > .5, alpha > .5
    core = interior(target)
    intersection = (fg & predicted).sum().item()
    union = (fg | predicted).sum().item()
    result = {'mask_l1': (alpha-target).abs().mean().item(),
              'mask_iou': intersection/max(union, 1),
              'interior_hole_rate': ((alpha < .5) & core).sum().item()/max(core.sum().item(), 1),
              'interior_pixels': core.sum().item()}
    if rgb is not None:
        mse = (rgb-gt).square()[:, fg[0]].mean().clamp_min(1e-12)
        result['foreground_psnr'] = (-10*torch.log10(mse)).item()
    return result


def local_affine_loss(canonical, deformed, anchors, neighbors, valid):
    """Fit a full affine map per neighborhood; do not penalize uniform stretch.

    Geometry and least-squares basis are detached. Only the deformation trajectory
    receives this regularizer's gradient. Degenerate neighborhoods are skipped.
    """
    x = canonical.detach()
    dx = x[neighbors]-x[anchors, None]
    dy = deformed[neighbors]-deformed[anchors, None]
    weights = valid.to(dx.dtype)[..., None]
    dx = dx*weights
    dy = dy*weights
    with torch.no_grad():
        singular = torch.linalg.svdvals(dx)
        usable = (valid.sum(-1) >= 4) & (singular[:, -1] > 1e-4*singular[:, 0].clamp_min(1e-8))
        basis = torch.linalg.pinv(dx, rtol=1e-4)
        scale = dx.square().sum(-1).sum(-1)/valid.sum(-1).clamp_min(1)
    fit = dx @ (basis @ dy)
    residual = (dy-fit).square().sum(-1)*valid
    per_anchor = residual.sum(-1)/valid.sum(-1).clamp_min(1)/scale.clamp_min(1e-8)
    if not usable.any():
        return deformed.sum()*0
    return per_anchor[usable].mean()


@torch.no_grad()
def neighborhoods(x, iteration, count=128, k=8, radius=.1, object_ids=None):
    n = len(x)
    count, k = min(count, n), min(k, n-1)
    if k < 4:
        raise ValueError('At least five Gaussians are required')
    anchors = (torch.arange(count, device=x.device)*n//count+iteration//10) % n
    # Bound peak memory; no RNG consumption and no cross-object edges.
    out, gates = [], []
    for chunk in anchors.split(32):
        distance = torch.cdist(x[chunk], x)
        distance.scatter_(1, chunk[:, None], float('inf'))
        if object_ids is not None:
            distance.masked_fill_(object_ids[chunk, None] != object_ids[None], float('inf'))
        values, index = distance.topk(k, largest=False)
        out.append(index); gates.append(values < radius)
    return anchors, torch.cat(out), torch.cat(gates)
