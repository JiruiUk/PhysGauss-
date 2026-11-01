"""Train-clock interpolation diagnostics.

All interpolation knots must come from training timestamps. No RGB, mask,
validation timestamp or object-instance target is accepted by these functions.
"""
import bisect
import math
import torch
import torch.nn.functional as F
from variants.kinematic_tuning.model import center_trajectory


def bracket(times,time):
    if len(times)<2 or any(not math.isfinite(t) for t in times) or any(a>=b for a,b in zip(times,times[1:])):
        raise ValueError('Need strictly increasing finite training times')
    if not math.isfinite(time):raise ValueError('Nonfinite query time')
    j=bisect.bisect_left(times,time)
    if (j<len(times) and abs(times[j]-time)<1e-6) or (j and abs(times[j-1]-time)<1e-6):return None
    if j==0 or j==len(times):return None  # Never extrapolate beyond training support.
    left,right=times[j-1],times[j]
    return left,right,(time-left)/(right-left)


def nlerp(a,b,fraction):
    a=F.normalize(a,dim=-1);b=F.normalize(b,dim=-1)
    b=torch.where((a*b).sum(-1,keepdim=True)<0,-b,b)
    return F.normalize((1-fraction)*a+fraction*b,dim=-1)


def raw_nlerp(a,b,fraction):
    # The released rasterizer consumes additive, nonunit rotation vectors.
    magnitude=(1-fraction)*a.norm(dim=-1,keepdim=True)+fraction*b.norm(dim=-1,keepdim=True)
    return nlerp(a,b,fraction)*magnitude


def repaired_state(x,canonical_scale,canonical_rotation,current,left,right,fraction,strength=0.,full=False):
    if strength==0:return current
    if not 0<=strength<=1 or not 0<=fraction<=1:raise ValueError('Invalid interpolation strength/fraction')
    result=dict(current)
    now=center_trajectory(x,current['d_xyz'])
    target=(1-fraction)*center_trajectory(x,left['d_xyz'])+fraction*center_trajectory(x,right['d_xyz'])
    # Retain current warp quaternion; change b to obtain the desired world center.
    result['d_xyz']=torch.cat((current['d_xyz'][...,:4],current['d_xyz'][...,4:]+strength*(target-now)),dim=-1)
    if full:
        target_scale=(1-fraction)*(canonical_scale+left['d_scaling']).abs()+fraction*(canonical_scale+right['d_scaling']).abs()
        current_scale=(canonical_scale+current['d_scaling']).abs()
        result['d_scaling']=(1-strength)*current_scale+strength*target_scale-canonical_scale
        target_rotation=raw_nlerp(canonical_rotation+left['d_rotation'],canonical_rotation+right['d_rotation'],fraction)
        result['d_rotation']=raw_nlerp(canonical_rotation+current['d_rotation'],target_rotation,strength)-canonical_rotation
    return result


def interpolate_query(query,x,time,training_times,canonical_scale,canonical_rotation,strength=0.,full=False):
    current=query(x,time)
    if strength==0:return current
    bounds=bracket(training_times,float(time))
    if bounds is None:return current
    left,right,fraction=bounds
    return repaired_state(x,canonical_scale,canonical_rotation,current,query(x,left),query(x,right),fraction,strength,full)


