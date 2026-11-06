"""Joint candidate and declared read-only controls; no training or GT."""
from collections import Counter
import numpy as np
from . import core, io
from ..i10_pseudo_depth.core import depth_at,in_hull,boundary,backproject,motion_check
from ..i10_pseudo_depth.audit import clean


def downstream(calibration,raw,p):
    old=io.read(io.ROOT/p['old_audit']);motion=[];targets={}
    parent=io.read(io.ROOT/p['old_protocol']);asset=np.load(io.ROOT/parent['trajectories'])
    for q in old['motion']:
        if 'row' not in q:
            motion.append(dict(id=q['id'],start=q['start'],passed=False,reason=q['reason']));continue
        start=q['start'];uv=np.array(q['uv']);z=np.full(12,np.nan);valid=np.zeros(12,bool);reasons=[]
        for t in range(12):
            i=start+t;c=calibration[i];reason='uncalibrated'
            if c['passed']:
                inside=in_hull(uv[t:t+1],np.array(c['uv'])[c['fit_ids']])[0]
                contrast,bgood=boundary(raw[i],uv[t:t+1],c['coefficients'],p['boundary_radius'])
                z[t]=depth_at(raw[i],uv[t:t+1],c['coefficients'])[0]
                valid[t]=inside and bgood[0] and contrast[0]<=p['boundary_relative_jump'] and np.isfinite(z[t])
                reason='passed_local_interpolation' if valid[t] else ('outside_fit_hull' if not inside else 'boundary_or_invalid')
            reasons.append(reason)
        world=backproject(uv,z,asset['views'][start:start+12],asset['k'])
        result=motion_check(world,np.array(q['parent_world']),valid,z,p['motion'])
        record=dict(id=q['id'],start=start,row=q['row'],uv=q['uv'],prior_depth=z.tolist(),
            local_eligible=valid.tolist(),local_reasons=reasons,**result)
        motion.append(record)
        if result['passed']:
            for t in result['fit_ids']:
                i=start+t;c=calibration[i]
                confidence=max(0.,1-c['check_P90_relative']/.1)*max(0.,1-result['motion_P90_relative']/.05)
                targets.setdefault(str(i),[]).append(dict(id=q['id'],row=q['row'],uv=uv[t].tolist(),depth=z[t],confidence=confidence))
    groups={k:dict(observations=0,learned_visibility_failure=0,bidirectional_disagreement=0) for k in ['boundary_proxy','interior_proxy','unknown']}
    for start in p['starts']:
        with np.load(io.ROOT/f'experiments/staged_reconstruction/RubberDuck_I10_tracking/dense/v1/clip_{start:03d}.npz') as f:
            for t in range(12):
                i=start+t;uv=f['forward'][t];c=calibration[i];labels=np.full(len(uv),'unknown',object)
                inside=np.isfinite(uv).all(1)&(uv[:,0]>=2)&(uv[:,0]<=397)&(uv[:,1]>=2)&(uv[:,1]<=222)
                if c['passed']:
                    ids=np.flatnonzero(inside);contrast,good=boundary(raw[i],uv[ids],c['coefficients'])
                    labels[ids[good]]='interior_proxy';labels[ids[good&(contrast>.1)]]='boundary_proxy'
                failure=~(f['visible'][t]&f['reverse_visible'][t]);disagree=np.linalg.norm(f['forward'][t]-f['reverse'][t],axis=1)>.75
                for key in groups:
                    selected=labels==key;g=groups[key];g['observations']+=int(selected.sum())
                    g['learned_visibility_failure']+=int((selected&failure).sum())
                    g['bidirectional_disagreement']+=int((selected&disagree).sum())
    assert sum(g['observations'] for g in groups.values())==53196
    good=[q for q in motion if q['passed']];cgood=[i for i,c in enumerate(calibration) if c['passed']]
    cb=[sum(i//40==b for i in cgood) for b in range(3)];mb=[sum(q['start']//40==b for q in good) for b in range(3)]
    distinct=len({q['row'] for q in good});g=p['local_gate']
    checks=dict(calibrated_views=len(cgood)>=g['calibrated_views'],calibrated_views_per_third=min(cb)>=g['calibrated_views_per_third'],
        motion_tracks=len(good)>=g['motion_tracks'],motion_tracks_per_third=min(mb)>=g['motion_tracks_per_third'],distinct_rows=distinct>=g['distinct_rows'])
    reasons=Counter(s for q in motion if 'local_reasons' in q for s in q['local_reasons'])
    summary=dict(calibrated_views=len(cgood),calibrated_views_by_third=cb,calibration_reasons=dict(Counter(c['reason'] for c in calibration)),
        motion_native_hypotheses=122,all_motion_rows=len(motion),motion_numeric_test_reached=sum(q.get('reason')!='insufficient_observations' for q in motion if 'row' in q),
        reliable_motion_tracks=len(good),reliable_tracks_by_third=mb,distinct_rows=distinct,
        observation_reasons=dict(reasons),auxiliary_observations=sum(map(len,targets.values())),
        gate_checks=checks,local_evidence_gate_passed=all(checks.values()),optimizer_updates=0,
        maximum_valid_observations=max(q.get('valid',0) for q in motion))
    return dict(summary=summary,motion=motion,targets=targets,boundary_groups=groups)


def calculate():
    p=io.read(io.BASE/'protocol.json');op=io.read(io.ROOT/p['old_protocol'])
    original=io.read(io.ROOT/op['tracks'])['rows']
    stationary=[dict(q,id='old_'+q['id']) for q in original if q['stationarity']['label']=='stationary_compatible']
    extraction=[]
    for start in p['starts']:
        q=io.read(io.RUNS/'clips'/f'{start:03d}.json');extraction.append({k:q[k] for k in ['start','seeds','qualified']})
        stationary.extend(r for r in q['rows'] if r['stationarity']['label']=='stationary_compatible')
    asset=np.load(io.ROOT/op['trajectories']);bank,grouping=core.anchor_bank(stationary,asset['views'],asset['k'],p['anchor_groups'])
    raw=[]
    for i in range(120):
        with np.load(io.ROOT/p['raw_maps']/f'{i:03d}.npz') as f:raw.append(f['relative'].copy())
    norms=[core.normalize(q) for q in raw];solutions=[]
    for half in [None,0,1]:solutions.append(core.joint_fit(bank,norms,raw,p['joint'],half,enabled=True))
    cal=[];control=[]
    for i in range(120):
        assert not io.stopped()
        co=[q['coefficients'][i] if q['coefficients'] is not None else None for q in solutions]
        cal.append(core.check_frame(raw[i],bank[i],co[0],co[1:],p['calibration'],all(q['success'] for q in solutions)))
        control.append(core.framewise(raw[i],bank[i],p['calibration']))
    return clean(dict(candidate=dict(calibration=cal,**downstream(cal,raw,p)),
        framewise_control=dict(calibration=control,**downstream(control,raw,p)),joint_solutions=solutions,
        anchor_bank=bank,grouping=grouping,extraction=extraction,
        stationary_tracks=len(stationary),original_stationary_tracks=sum(q['id'].startswith('old_') for q in stationary),
        expanded_stationary_tracks=sum(q['id'].startswith('lk_') for q in stationary),
        normalizations=norms,optimizer_updates=0,independent_geometry_GT=False))
