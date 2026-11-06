"""One train-only CPU decision pass, all denominators and unknowns preserved."""
from collections import Counter
import numpy as np
from .io import ROOT,BASE,REPORT,RUNS,read,write,sha,stopped
from .core import calibrate,depth_at,in_hull,boundary,backproject,motion_check
from ..i10_tracking.core import project_world


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.bool_):return bool(v)
    return v


def calculate():
    p=read(BASE/'protocol.json');tracks=read(ROOT/p['tracks'])['rows']
    original={q['id']:q for q in tracks};native=read(ROOT/p['visibility'])['rows']
    asset=np.load(ROOT/p['trajectories']);views,k,xyz=[asset[n] for n in ['views','k','xyz']]
    raw=[];anchors=[[] for _ in range(120)]
    for q in tracks:
        if q['stationarity']['label']!='stationary_compatible':continue
        start=q['start'];point=np.array(q['stationarity']['stationary_hypothesis_xyz'])
        _,z=project_world(point[None],views[start:start+12],k)
        for t,uv in enumerate(q['uv']):
            anchors[start+t].append(dict(id=q['id'],uv=uv,depth=float(z[t,0])))
    calibration=[]
    for i in range(120):
        assert not stopped()
        with np.load(RUNS/'raw'/f'{i:03d}.npz') as f:raw.append(f['relative'].copy())
        calibration.append(calibrate(raw[-1],anchors[i],p['calibration']))
    motion=[];eligible={};all_observations=0
    for q in native:
        if not (q['binding']['accepted'] and q['reciprocal']['passed'] and q['visibility']['passed']):
            motion.append(dict(id=q['id'],start=q['start'],passed=False,reason='original_native_binding_not_admitted'));continue
        parentrow=q['binding']['row'];start=q['start'];uv=np.array(q['uv']);z=np.full(12,np.nan)
        local=np.zeros(12,bool);jumps=np.full(12,np.nan);local_reasons=[]
        for t in range(12):
            i=start+t;c=calibration[i];reason='uncalibrated'
            if c['passed']:
                z[t]=depth_at(raw[i],uv[t:t+1],c['coefficients'])[0]
                inside=in_hull(uv[t:t+1],np.array(c['uv'])[c['fit_ids']])[0]
                contrast,valid=boundary(raw[i],uv[t:t+1],c['coefficients'],p['boundary_radius'])
                jumps[t]=contrast[0]
                local[t]=inside and valid[0] and contrast[0]<=p['boundary_relative_jump'] and np.isfinite(z[t])
                reason='passed_local_interpolation' if local[t] else ('outside_fit_hull' if not inside else 'boundary_or_invalid')
            local_reasons.append(reason)
        parent=xyz[start:start+12,parentrow].astype(float)
        proposed=backproject(uv,z,views[start:start+12],k)
        checked=motion_check(proposed,parent,local,z,p['motion'])
        record=dict(id=q['id'],start=start,row=parentrow,band=start//40,**checked,
            uv=uv.tolist(),prior_depth=z.tolist(),local_eligible=local.tolist(),local_reasons=local_reasons,
            boundary_contrast=jumps.tolist(),prior_world=proposed.tolist(),parent_world=parent.tolist(),
            camera_z_relative_span=float(np.ptp(z[np.isfinite(z)])/np.median(z[np.isfinite(z)])) if np.isfinite(z).any() else None,
            parent_motion_not_GT=True)
        motion.append(record);all_observations+=12
        if checked['passed']:
            for t in np.flatnonzero(local):
                i=start+int(t)
                if i%2:continue # odd images remain RGB observations, not auxiliary targets
                c=calibration[i]
                confidence=max(0.,1-c['check_P90_relative']/p['calibration']['relative_P90_max'])*max(0.,1-checked['motion_P90_relative']/p['motion']['motion_P90_max'])
                eligible.setdefault(str(i),[]).append(dict(id=q['id'],row=parentrow,uv=uv[t].tolist(),depth=float(z[t]),confidence=float(confidence)))
    # All original proposals, including invisible/failed trajectories. Never
    # equate tracker failure or discontinuity with a true occlusion label.
    groups={key:dict(observations=0,learned_visibility_failure=0,bidirectional_disagreement=0,track_rejected=0)
            for key in ['boundary_proxy','interior_proxy','unknown']}
    boundary_frames=[];proposals=0
    for start in range(0,109,4):
        f=np.load(ROOT/f'experiments/staged_reconstruction/RubberDuck_I10_tracking/dense/v1/clip_{start:03d}.npz')
        proposals+=len(f['seeds'])
        for t in range(12):
            i=start+t;uv=f['forward'][t];finite=np.isfinite(uv).all(1)
            inside=finite&(uv[:,0]>=2)&(uv[:,0]<=397)&(uv[:,1]>=2)&(uv[:,1]<=222)
            known=np.zeros(len(uv),bool);contrast=np.full(len(uv),np.nan);c=calibration[i]
            if c['passed']:
                ids=np.flatnonzero(inside)
                values,valid=boundary(raw[i],uv[ids],c['coefficients'],p['boundary_radius'])
                known[ids]=valid;contrast[ids]=values
            labels=np.full(len(uv),'unknown',dtype=object)
            labels[known]='interior_proxy';labels[known&(contrast>p['boundary_relative_jump'])]='boundary_proxy'
            visibility=~(f['visible'][t].astype(bool)&f['reverse_visible'][t].astype(bool))
            inconsistent=np.linalg.norm(f['forward'][t]-f['reverse'][t],axis=1)>.75
            for key in groups:
                select=labels==key;n=int(select.sum());g=groups[key];g['observations']+=n
                g['learned_visibility_failure']+=int((select&visibility).sum())
                g['bidirectional_disagreement']+=int((select&inconsistent).sum())
                g['track_rejected']+=int((select&~f['accepted'].astype(bool)).sum())
            boundary_frames.append(dict(start=start,frame=i,total=len(uv),known=int(known.sum()),
                boundary=int((labels=='boundary_proxy').sum()),interior=int((labels=='interior_proxy').sum()),unknown=int((labels=='unknown').sum())))
    assert proposals==4433 and sum(g['observations'] for g in groups.values())==4433*12
    validframes=[i for i,c in enumerate(calibration) if c['passed']]
    passed=[q for q in motion if q['passed']];bands=[sum(q['start']//40==b for q in passed) for b in range(3)]
    cbands=[sum(i//40==b for i in validframes) for b in range(3)];distinct=len({q['row'] for q in passed})
    gate=p['local_gate'];checks=dict(calibrated_views=len(validframes)>=gate['calibrated_views'],
        calibrated_views_per_third=min(cbands)>=gate['calibrated_views_per_third'],
        motion_tracks=len(passed)>=gate['motion_tracks'],motion_tracks_per_third=min(bands)>=gate['motion_tracks_per_third'],
        distinct_rows=distinct>=gate['distinct_rows'])
    summary=dict(calibrated_views=len(validframes),calibrated_views_by_third=cbands,
        calibration_reasons=dict(Counter(q['reason'] for q in calibration)),
        motion_native_hypotheses=sum('row' in q for q in motion),all_motion_rows=len(motion),
        reliable_motion_tracks=len(passed),reliable_tracks_by_third=bands,distinct_rows=distinct,
        auxiliary_observations=sum(len(q) for q in eligible.values()),auxiliary_frames=len(eligible),
        auxiliary_observations_by_third=[sum(len(q) for f,q in eligible.items() if int(f)//40==b) for b in range(3)],
        boundary_original_proposals=proposals,boundary_observation_denominator=4433*12,
        gate_checks=checks,local_evidence_gate_passed=all(checks.values()),
        original_T4_global_failure_preserved=True,independent_geometry_GT=False,optimizer_updates=0)
    return clean(dict(summary=summary,calibration=calibration,motion=motion,eligible_fit_observations=eligible,
        boundary_groups=groups,boundary_window_frames=boundary_frames,
        raw_sha256={str(i):sha(RUNS/'raw'/f'{i:03d}.npz') for i in range(120)}))
