"""Signed-distance nonpenetration, explicitly gated by observational eligibility."""
import torch
import torch.nn.functional as F


def sample_sdf(points,field,origin,spacing):
    """World xyz to xyz-volume samples; points outside the grid are separated.

    The input volume is a conditional reconstructed static solid hypothesis,
    not a certified contact surface. Trilinear derivatives are piecewise linear.
    """
    origin=torch.as_tensor(origin,dtype=points.dtype,device=points.device)
    size=points.new_tensor(field.shape);maximum=origin+(size-1)*spacing
    xyz=(points-origin)/spacing
    normalized=2*xyz/(size-1)-1
    grid=normalized[:,[2,1,0]][None,None,None]
    values=F.grid_sample(field[None,None],grid,mode='bilinear',padding_mode='border',align_corners=True).flatten()
    clipped=torch.maximum(torch.minimum(points,maximum),origin)
    outside=((points<origin)|(points>maximum)).any(-1)
    return torch.where(outside,(points-clipped).norm(dim=-1)+spacing,values)


def nonpenetration(distance=None,eligible=None,margin=0.,enabled=False):
    if not enabled:return None
    if margin<0 or distance.ndim!=1 or eligible.shape!=distance.shape or eligible.dtype!=torch.bool:
        raise ValueError('Expected signed distances and explicit eligible mask')
    if not bool(torch.isfinite(distance[eligible]).all()):raise ValueError('Nonfinite eligible distance')
    selected=distance[eligible];depth=(-selected-margin).clamp_min(0)
    loss=depth.square().mean() if len(selected) else selected.sum()*0
    return loss,{'eligible_count':len(selected),'active_violations':int((depth>0).sum()),
        'max_proxy_penetration':float(depth.detach().max()) if len(depth) else None,
        'no_eligible_samples_is_not_collision_validation':not len(selected),
        'distance_is_conditional_geometry_not_GT':True}
