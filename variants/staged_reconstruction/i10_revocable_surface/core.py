"""Detached RGB evidence, persistent patch hypotheses, reversible constraints."""
from copy import deepcopy
import numpy as np


def build_patches(evidence, xyz, projected, config):
    """Fit alone selects members/weights. Held observations only validate later."""
    patches=[];fit=config['fit']
    for t in evidence['rows']:
        if not t['accepted']:continue
        sl=slice(t['start'],t['start']+12);x=xyz[sl];uv=projected[sl]
        anchor=t['binding']['row'];eligible=[]
        for row in t['binding']['eligible_rows']:
            seen=t['visibility_by_row'][str(row)]
            if sum(seen[k]['passed'] for k in fit)<config['min_fit_visible']:continue
            if (not np.isfinite(x[fit,row]).all() or
                np.linalg.norm(x[fit,row]-x[fit,anchor],axis=-1).max()>config['patch_world_distance'] or
                np.linalg.norm(uv[fit,row]-uv[fit,anchor],axis=-1).max()>config['patch_projection_distance']):continue
            score=float(np.mean(np.sum((uv[fit,row]-np.asarray(t['uv'])[fit])**2,axis=-1)))
            eligible.append((score,int(row)))
        rows=[anchor]+[row for _,row in sorted(eligible) if row!=anchor][:config['patch_max_rows']-1]
        masses=[np.mean([t['visibility_by_row'][str(row)][k]['numerator'] for k in fit]) for row in rows]
        assert min(masses)>=0 and sum(masses)>0
        weights=np.asarray(masses,np.float64)/sum(masses)
        centers=np.einsum('tnc,n->tc',x[:,rows],weights)
        patches.append(dict(id=t['id'],start=t['start'],band=t['band'],uv=t['uv'],
            rows=rows,weights=weights.tolist(),parent_centers=centers.tolist(),
            fit_only_selection=True,physical_identity=False))
    return sorted(patches,key=lambda q:q['id'])


def geometry(lengths, indices, config):
    values=np.asarray(lengths)[indices]
    return bool(np.isfinite(values).all() and values.min()>=config['edge_min_distance'] and
                values.max()<=config['edge_max_distance'] and
                np.ptp(values)/max(float(values.mean()),1e-12)<=config['edge_relative_length_range_max'])


def build_edges(patches,config):
    proposed={};fit=config['fit']
    for i,a in enumerate(patches):
        options=[]
        for j,b in enumerate(patches):
            if i==j or a['start']!=b['start'] or set(a['rows'])&set(b['rows']):continue
            lengths=np.linalg.norm(np.asarray(a['parent_centers'])-b['parent_centers'],axis=-1)
            if geometry(lengths,fit,config):options.append((float(lengths[fit].mean()),j,lengths))
        for _,j,lengths in sorted(options,key=lambda v:(v[0],v[1]))[:config['edge_neighbor_limit']]:
            left,right=sorted([i,j]);proposed[(left,right)]=dict(i=left,j=right,start=a['start'],
                parent_lengths=lengths.tolist(),fit_selected=True)
    return [proposed[key] for key in sorted(proposed)]


def patch_measurement(projected, depth, visibility, appearance, target, config):
    error=np.linalg.norm(np.asarray(projected)-target,axis=-1)
    fit,held=config['fit'],config['held']
    finite=bool(np.isfinite(error).all() and np.isfinite(depth).all())
    fit_rms=float(np.sqrt(np.mean(error[fit]**2))) if finite else None
    held_rms=float(np.sqrt(np.mean(error[held]**2))) if finite else None
    frames=[];contradiction=False
    for k,(v,a) in enumerate(zip(visibility,appearance)):
        textured=min(a['observed_std'],a['rendered_std'])>=config['texture_std_255_min']
        known=bool(v['known'] and v['passed'] and textured and np.isfinite(error[k]) and depth[k]>config['depth_min'])
        passed=bool(known and a['ncc']>=config['ncc_min'] and error[k]<=config['projection_max'])
        contradiction|=bool(known and not passed)
        frames.append(dict(passed=passed,known=known,error_px=float(error[k]) if np.isfinite(error[k]) else None,
            share=v['share'],ncc=float(a['ncc']),observed_std=float(a['observed_std']),rendered_std=float(a['rendered_std'])))
    good=np.array([v['passed'] for v in frames])
    geometric=bool(finite and np.asarray(depth).min()>config['depth_min'] and fit_rms<=config['projection_rms_max'] and
                   held_rms<=config['projection_rms_max'] and error.max()<=config['projection_max'])
    passed=bool(geometric and good[fit].sum()>=config['min_fit_visible'] and
                good.sum()>=config['min_valid_frames'] and good[held].sum()>=config['min_valid_held'])
    contradiction|=bool(finite and not geometric and any(v['known'] for v in frames))
    confidence=(min(float(good[fit].mean()),float(good[held].mean()))*
                max(0.,1-max(fit_rms,held_rms)/config['projection_max'])) if passed else 0.
    return dict(passed=passed,kind='pass' if passed else ('contradiction' if contradiction else 'unknown'),
                confidence=confidence,frames=frames,fit_rms_px=fit_rms,held_rms_px=held_rms)


def transition(previous,measurement,config,initial=False):
    old=previous or dict(status='unknown',pass_streak=0,fail_streak=0,confidence=0.)
    new=deepcopy(old);kind=measurement['kind']
    if kind=='pass':
        new['pass_streak']=old['pass_streak']+1;new['fail_streak']=0
        active=initial or old['status']=='active' or new['pass_streak']>=config['passes_to_restore']
        new['status']='active' if active else old['status']
        new['confidence']=measurement['confidence'] if active else 0.
    else:
        new['pass_streak']=0;new['confidence']=0.
        new['fail_streak']=old['fail_streak']+1 if kind=='contradiction' else 0
        new['status']='revoked' if old['status']=='revoked' or new['fail_streak']>=config['failures_to_revoke'] else 'suspended'
    new['measurement']=measurement
    return new


def edge_measurement(lengths,left,right,config):
    endpoints=left['status']==right['status']=='active'
    valid=geometry(lengths,config['fit'],config) and geometry(lengths,config['held'],config)
    return dict(kind='unknown' if not endpoints else ('pass' if valid else 'contradiction'),passed=bool(endpoints and valid),
        confidence=min(left['confidence'],right['confidence']) if endpoints and valid else 0.,
        fit_geometry=geometry(lengths,config['fit'],config),held_geometry=geometry(lengths,config['held'],config))


def add_constraints(base,world,view,k,frame,patches,edges,states,edge_states,config,enabled=False):
    if not enabled:return base,base*0,base*0,dict(tracks=0,edges=0)
    import torch
    centers=[]
    for q in patches:
        ids=torch.tensor(q['rows'],device=world.device,dtype=torch.long)
        centers.append((world[ids]*world.new_tensor(q['weights'])[:,None]).sum(0))
    if not centers:return base,base*0,base*0,dict(tracks=0,edges=0)
    centers=torch.stack(centers);camera=centers@view[:3,:3].T+view[:3,3]
    q=camera@k.T;uv=q[:,:2]/q[:,2:3].clamp_min(1e-8)
    track=base*0;local=base*0;nt=ne=dt=de=0;usable={}
    for index,(patch,state) in enumerate(zip(patches,states)):
        t=frame-patch['start']
        if t not in config['fit']:continue
        dt+=1
        if state['status']!='active' or not state['measurement']['frames'][t]['passed']:continue
        error=uv[index]-uv.new_tensor(patch['uv'][t])
        valid=bool(torch.isfinite(error).all() and camera[index,2].detach()>config['depth_min'] and
                   torch.linalg.vector_norm(error.detach())<=config['projection_max'])
        if not valid:continue
        usable[index]=state['confidence'];nt+=1
        track=track+state['confidence']*(torch.sqrt(1+error.square().sum())-1)
    for edge,state in zip(edges,edge_states):
        t=frame-edge['start']
        if t not in config['fit']:continue
        de+=1;i,j=edge['i'],edge['j']
        if state['status']!='active' or i not in usable or j not in usable:continue
        length=torch.linalg.vector_norm(centers[j]-centers[i])
        if not bool(torch.isfinite(length) and config['edge_min_distance']<=length.detach()<=config['edge_max_distance']):continue
        confidence=min(state['confidence'],usable[i],usable[j]);ne+=1
        error=(length-edge['parent_lengths'][t])/config['local_scale']
        local=local+confidence*(torch.sqrt(1+error.square())-1)
    track=config['track_weight']*track/max(dt,1);local=config['local_weight']*local/max(de,1)
    return base+track+local,track,local,dict(tracks=nt,edges=ne,track_denominator=dt,edge_denominator=de)
