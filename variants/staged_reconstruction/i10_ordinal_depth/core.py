"""Detached ordinal evidence and default-off robust near/far loss."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from ..i10_pseudo_depth.core import depth_at,in_hull,boundary
from ..i10_tracking.core import sample


def make_pairs(a,raw,p,held=False):
    proposals=[];pairs=[]
    for track in a['candidate']['motion']:
        if not track['passed']:continue
        times=track['held_ids'] if held else track['fit_ids']
        for t in times:
            index=track['start']+t;c=a['candidate']['calibration'][index];u=np.array(track['uv'][t])
            anchors=[q for q in a['anchor_bank'][index] if (q['split']=='check')==held]
            accepted=0
            for anchor in sorted(anchors,key=lambda q:(q['group'],q['id'])):
                v=np.array(anchor['uv']);uv=np.array([u,v]);reason='passed';sign=0
                if np.linalg.norm(u-v)<p['minimum_pixel_separation']:reason='nearby_pixels'
                elif not in_hull(uv,np.array(c['uv'])[c['fit_ids']]).all():reason='outside_fit_hull'
                else:
                    _,spread=a['normalizations'][index];rv=sample(raw[index],uv).astype(float)
                    if abs(rv[0]-rv[1])/spread<p['raw_normalized_gap_min']:reason='weak_raw_order'
                    else:
                        values=np.array([depth_at(raw[index],uv,co) for co in [c['coefficients']]+c['split_coefficients']])
                        contrast,valid=boundary(raw[index],uv,c['coefficients'])
                        if not np.isfinite(values).all() or not valid.all() or (contrast>p['maximum_prior_boundary_jump']).any():reason='boundary_or_invalid'
                        else:
                            logs=np.log(values);margin=math.log(p['margin_ratio'])
                            if logs[:,0].max()+margin<logs[:,1].min():sign=1
                            elif logs[:,1].max()+margin<logs[:,0].min():sign=-1
                            else:reason='overlapping_calibration_intervals'
                row=dict(id=f'{track["id"]}:{index}:{anchor["id"]}',track=track['id'],frame=index,row=track['row'],
                    anchor=anchor['id'],anchor_group=anchor['group'],uv=[u.tolist(),v.tolist()],sign=sign,reason=reason,
                    confidence=max(0.,1-track['motion_P90_relative']/.05),auxiliary_held=held)
                if reason=='passed':
                    if accepted>=p['maximum_pairs_per_frame']:row['reason']='fixed_pair_cap'
                    else:pairs.append(row.copy());accepted+=1
                proposals.append(row)
    return dict(pairs=pairs,proposals=proposals,held=held)


def sampled_depth(package,uv):
    D=package['depth'];alpha=package['alpha'];h,w=D.shape[-2:]
    points=torch.as_tensor(uv,device=D.device,dtype=D.dtype)
    grid=(points/points.new_tensor([w-1,h-1])*2-1).reshape(1,1,-1,2)
    s=lambda image:F.grid_sample(image.reshape(1,1,h,w),grid,align_corners=True).reshape(-1)
    d=s(D);a=s(alpha);return d/a.clamp_min(1e-8),a


def ranking(base,z=None,sign=None,confidence=None,valid=None,p=None,enabled=False):
    if not enabled:return base,base*0 if isinstance(base,torch.Tensor) else None
    if z.numel()==0:return base,z.sum()*0
    sign=sign.detach();confidence=confidence.detach();valid=valid.detach()
    safe=torch.where(valid[:,None],z,torch.ones_like(z))
    gap=sign*(torch.log(safe[:,1])-torch.log(safe[:,0]))
    violation=torch.relu(math.log(p['margin_ratio'])-gap)
    energy=torch.sqrt(1+(violation/p['log_error_scale']).square())-1
    aux=p['weight']*(energy*confidence*valid).sum()/len(z)
    return base+aux,aux


def transition(previous,known,passed,initial=False):
    state=dict(previous or dict(status='suspended',failures=0,passes=0))
    if not known:state.update(status='suspended',passes=0);return state
    if passed:
        state.update(passes=state['passes']+1,failures=0)
        state['status']='active' if initial or state['passes']>=2 else 'suspended'
    else:
        state.update(failures=state['failures']+1,passes=0)
        state['status']='revoked' if state['failures']>=2 else 'suspended'
    return state
