"""Fixed persistent IDs, no semantic selectors or per-frame nearest matching."""
import numpy as np


def observations(evidence):
    bins={}
    for track in evidence['rows']:
        binding=track.get('binding',{})
        if not binding.get('accepted'):continue
        for t,uv in enumerate(track['uv']):
            key=(track['start']+t,binding['row'])
            bins.setdefault(key,[]).append(uv)
    result={}
    for (frame,row),values in sorted(bins.items()):
        result.setdefault(frame,[]).append((row,np.mean(values,axis=0).tolist()))
    return result


def add_tracks(rgb_loss,scene,cam,t,entries,enabled=False,weight=.01):
    if not enabled:return rgb_loss,rgb_loss*0
    import torch
    from ..i10_tracking.core import tracking_objective
    if not entries:return rgb_loss,rgb_loss*0
    all_xyz=scene.dynamic_at(t).get_xyz
    rows=torch.tensor([a[0] for a in entries],device=all_xyz.device,dtype=torch.long)
    target=all_xyz.new_tensor([a[1] for a in entries])[None]
    focal=400/(2*np.tan(cam.FoVx/2))
    k=all_xyz.new_tensor([[focal,0,199.5],[0,focal,112],[0,0,1]])
    zero=rgb_loss*0
    term=tracking_objective(zero,all_xyz[rows][None],cam.world_view_transform.T[None],k,target,True,weight)
    return rgb_loss+term,term
