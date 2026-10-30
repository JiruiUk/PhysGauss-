"""Opt-in predicted mean-RGB feature block; not semantic supervision."""
import numpy as np
from variants.object_centers.motion_grouping import motion_features


def color_block(colors,count):
    colors=np.asarray(colors,dtype=np.float32)
    if colors.shape!=(count,3) or not np.isfinite(colors).all() or (colors<0).any() or (colors>1).any():
        raise ValueError('Finite aligned RGB[N,3] in [0,1] required')
    return colors


def motion_color_features(poses,times,mode,colors,color_weight=1.):
    if not np.isfinite(color_weight) or color_weight<0:raise ValueError('Nonnegative finite color weight required')
    motion=motion_features(poses,times,mode)
    if color_weight==0:return motion
    return np.concatenate([motion,color_block(colors,poses.shape[1])*color_weight],axis=1).astype(np.float32)
