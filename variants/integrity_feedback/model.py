import math
import torch


def foreground_tail_loss(image,target,mask,fraction=.2):
    """Emphasize the worst TRAINING foreground RGB pixels; no new image labels."""
    if not 0<fraction<=1:raise ValueError('Tail fraction must be in (0,1]')
    pixels=(image-target.detach()).abs().mean(0)[mask.detach()[0]>.5]
    if not len(pixels):return image.sum()*0
    return pixels.topk(max(1,math.ceil(fraction*len(pixels)))).values.mean()
