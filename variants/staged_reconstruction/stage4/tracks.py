"""Sparse raw-image tracks, with no inferred Gaussian/material IDs."""
import torch
from variants.flow_comparison.geometry import sample,forward_backward_valid


def advect(points,forward,backward,source_components,target_components,source_component,target_component):
    points=torch.as_tensor(points,dtype=forward.dtype,device=forward.device)
    if source_component<0 or target_component<0:return points,torch.zeros(len(points),dtype=torch.bool,device=points.device)
    query=points[:,None,:];delta=sample(forward,query)[:,0];target=points+delta
    back=sample(backward,target[:,None,:])[:,0]
    source_label=sample((source_components==source_component).to(forward.dtype),query)[:,0]
    target_label=sample((target_components==target_component).to(forward.dtype),target[:,None,:])[:,0]
    confidence=sample(forward_backward_valid(forward,backward).to(forward.dtype),query)[:,0]
    valid=(source_label>.99)&(target_label>.99)&(confidence>.99)&((delta+back).norm(dim=-1)<=.75)
    return target,valid
