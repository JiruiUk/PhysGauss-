"""Recurring raw prior orders, independently checked in time; no metric depths."""
import numpy as np
from ..i10_tracking.core import sample
from ..i10_ordinal_depth.core import ranking,sampled_depth,transition


def patch(raw,uv,spread,p):
    uv=np.array(uv,float);h,w=raw.shape
    if not np.isfinite(uv).all() or (uv<2).any() or uv[0]>w-3 or uv[1]>h-3 or spread<=1e-12:return None
    offsets=np.array([(x,y) for y in range(-2,3) for x in range(-2,3)])
    values=sample(raw,uv+offsets).astype(float)
    if not np.isfinite(values).all() or np.ptp(values)/spread>p['maximum_raw_patch_range']:return None
    return float(values.min()),float(values.max())


def make_pairs(a,raw,p,held=False):
    hypotheses=[];proposals=[]
    for track in a['candidate']['motion']:
        if 'row' not in track:continue # unchanged T4 native eligibility, not L7 metric eligibility
        start=track['start'];groups={}
        for t in range(12):
            for q in a['anchor_bank'][start+t]:
                if (q['split']=='check')!=held:continue
                groups.setdefault(q['group'],{}).setdefault(t,q)
        for group,observed in sorted(groups.items()):
            signs=np.zeros(12,int);gaps=np.full(12,np.nan)
            for t,anchor in observed.items():
                u=track['uv'][t];v=anchor['uv'];i=start+t
                if np.linalg.norm(np.array(u)-v)<p['minimum_pixel_separation']:continue
                spread=a['normalizations'][i][1];x=patch(raw[i],u,spread,p);y=patch(raw[i],v,spread,p)
                if x is None or y is None:continue
                if x[0]>y[1]+p['raw_normalized_gap_min']*spread:signs[t]=1;gaps[t]=(x[0]-y[1])/spread
                elif y[0]>x[1]+p['raw_normalized_gap_min']*spread:signs[t]=-1;gaps[t]=(y[0]-x[1])/spread
            fit=np.flatnonzero((signs!=0)&(np.arange(12)%2==0));check=np.flatnonzero((signs!=0)&(np.arange(12)%2==1))
            reason='passed';sign=0
            if len(fit)<p['minimum_order_fit_observations']:reason='insufficient_fit_orders'
            elif len(set(signs[fit]))!=1:reason='fit_order_conflict'
            else:
                sign=int(signs[fit[0]])
                if len(check)<p['minimum_order_check_observations']:reason='insufficient_check_orders'
                elif not (signs[check]==sign).all():reason='held_order_conflict'
            hypotheses.append(dict(track=track['id'],row=track['row'],start=start,anchor_group=group,
                signs=signs.tolist(),raw_gaps=[None if not np.isfinite(g) else float(g) for g in gaps],
                fit_count=len(fit),check_count=len(check),fit_chosen_sign=sign,reason=reason))
            if reason!='passed':continue
            for t in (check if held else fit):
                anchor=observed[int(t)];i=start+int(t)
                proposals.append(dict(id=f'{track["id"]}:{i}:{group}',track=track['id'],row=track['row'],frame=i,
                    anchor=anchor['id'],anchor_group=group,uv=[track['uv'][int(t)],anchor['uv']],
                    sign=sign,confidence=p['pair_confidence'],auxiliary_held=held,reason='passed'))
    chosen=[];counts={}
    for q in sorted(proposals,key=lambda q:(q['track'],q['anchor_group'],q['frame'])):
        n=counts.get(q['frame'],0)
        if n>=p['maximum_pairs_per_frame']:q['reason']='fixed_pair_cap'
        else:chosen.append(q.copy());counts[q['frame']]=n+1
    return dict(pairs=chosen,proposals=proposals,hypotheses=hypotheses,held=held)
