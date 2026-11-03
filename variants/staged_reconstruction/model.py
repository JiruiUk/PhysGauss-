"""Static-stage observation contract; unknown pixels never become black targets."""
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion


def observation_masks(rgba, dilation=3):
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    blue = (rgb[...,2] > rgb[...,0]*1.3+.02) & (rgb[...,2] > rgb[...,1]*1.05) & (alpha > .95)
    yellow = (np.minimum(rgb[...,0],rgb[...,1]) > rgb[...,2]*1.35+.03) & (rgb[...,0] > .15) & (rgb[...,1] > .13) & (alpha > .5)
    occluded = binary_dilation(yellow, iterations=max(1,dilation))
    positive = binary_erosion(blue, iterations=1) & ~occluded
    background = binary_erosion(alpha < .01, iterations=2, border_value=1) & ~occluded
    return positive, background


def masked_objective(rgb, alpha, target, positive, negative):
    if not bool(positive.any()):
        raise ValueError('No accepted static image support')
    a = alpha.squeeze(0)
    color = (rgb[:,positive]-target[:,positive]).abs().mean()
    foreground = (a[positive]-1).square().mean()
    background = a[negative].square().mean() if bool(negative.any()) else a.sum()*0
    return color + .2*foreground + .2*background, {'rgb':color, 'foreground':foreground, 'background':background}


def eligible_voxels(positive, background, seen, third_support):
    return ((positive >= 8) & (seen >= 12) & (background <= np.maximum(1, seen*.01))
            & ((third_support >= 2).sum(-1) >= 2))
