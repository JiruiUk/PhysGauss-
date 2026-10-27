"""Smooth positive radiance and foreground-balanced photometric objective."""
import math
import torch
import torch.nn.functional as F


def positive_color(raw,beta=5.):
    if not math.isfinite(beta) or beta<=0:raise ValueError('Positive finite beta required')
    return F.softplus(raw,beta=beta,threshold=20.)


def inverse_positive_color(color,beta=5.):
    if not math.isfinite(beta) or beta<=0 or not torch.isfinite(color).all() or (color<=0).any():
        raise ValueError('Inverse positive color requires finite positive inputs and beta')
    return color+torch.log(-torch.expm1(-beta*color))/beta


def foreground_rgb_loss(image,target,mask):
    if image.shape!=target.shape or image.ndim!=3 or image.shape[0]!=3 or mask.shape!=(1,*image.shape[1:]):
        raise ValueError('Expected RGB [3,H,W] and alpha [1,H,W]')
    mask=mask.detach()
    return ((image-target.detach()).abs()*mask).sum()/(3*mask.sum()).clamp_min(1.)
