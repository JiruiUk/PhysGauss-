import torch
from torch.nn import functional as F


def masked_mean(value, mask):
    if value.ndim > mask.ndim:
        mask = mask[..., None].expand_as(value)
    selected = value[mask.bool()]
    return selected.mean() if selected.numel() else value.sum()*0


def ssim_score(pred, target):
    """Small reference SSIM; not a substitute for an official benchmark evaluator."""
    x, y = pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    pool = lambda v: F.avg_pool2d(v, 3, stride=1)
    mx, my = pool(x), pool(y)
    vx, vy = (pool(x*x)-mx*mx).clamp_min(0), (pool(y*y)-my*my).clamp_min(0)
    covariance = pool(x*y)-mx*my
    return (((2*mx*my+0.01**2)*(2*covariance+0.03**2)) /
            ((mx*mx+my*my+0.01**2)*(vx+vy+0.03**2))).mean()


def visual_losses(rendered, frame):
    mask = frame.rgb_valid
    error = rendered.rgb-frame.rgb
    variance = rendered.variance[..., None]
    losses = {
        "rgb": masked_mean(error.abs(), mask),
        # SSIM is defined on the complete image only; exclude it on masked frames.
        "ssim": 1-ssim_score(rendered.rgb, frame.rgb) if bool(mask.all()) else error.sum()*0,
        "uncertainty": masked_mean(0.5*(error.square()/variance + variance.log()), mask),
        "uncertainty_prior": masked_mean((rendered.variance.log()-torch.log(variance.new_tensor(0.01))).square(), mask),
        "depth": error.sum()*0,
        "flow": error.sum()*0,
    }
    if frame.depth is not None:
        losses["depth"] = masked_mean((rendered.depth-frame.depth).abs(), frame.depth_valid)
    if frame.flow is not None and rendered.flow is not None:
        losses["flow"] = masked_mean(torch.linalg.vector_norm(rendered.flow-frame.flow, dim=-1), frame.flow_valid)
    return losses
