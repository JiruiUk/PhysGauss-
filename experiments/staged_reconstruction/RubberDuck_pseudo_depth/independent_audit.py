"""Separate arithmetic/accounting evaluator; no changes to frozen decisions."""
from pathlib import Path
from collections import Counter
import argparse
import cv2
import numpy as np
from variants.staged_reconstruction.i10_pseudo_depth import io


def samples(field,uv):
    p=np.asarray(uv,np.float32)
    return cv2.remap(field,p[:,0:1],p[:,1:2],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel().astype(float)


def affine(x,y):
    slopes=[];threshold=1e-6*max(1.,float(np.ptp(x)))
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            if abs(x[j]-x[i])>threshold:slopes.append((y[j]-y[i])/(x[j]-x[i]))
    if not slopes:return None
    a=float(np.median(slopes));b=float(np.median(y-a*x))
    return [a,b] if np.isfinite([a,b]).all() and a>0 else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-independent-audit',action='store_true')
    if not parser.parse_args().enable_independent_audit:raise SystemExit('Independent audit default off')
    io.verify();p=io.read(io.BASE/'protocol.json');a=io.read(io.REPORT/'audit.json')
    frozen=io.read(io.REPORT/'independent_evaluation_freeze.json');assert frozen['source_sha256']==io.sha(Path(__file__))
    data=np.load(io.ROOT/p['trajectories']);views,k,xyz=[data[n] for n in ['views','k','xyz']]
    station=[q for q in io.read(io.ROOT/p['tracks'])['rows'] if q['stationarity']['label']=='stationary_compatible']
    anchors=[[] for _ in range(120)];coefficients_checked=0;motion_checked=0
    for track in station:
        world=np.array(track['stationarity']['stationary_hypothesis_xyz'])
        for t,uv in enumerate(track['uv']):
            i=track['start']+t;z=float(views[i,2,:3]@world+views[i,2,3])
            anchors[i].append(dict(id=track['id'],uv=uv,depth=z))
    raw=[]
    for i,c in enumerate(a['calibration']):
        with np.load(io.RUNS/'raw'/f'{i:03d}.npz') as f:field=f['relative'].copy()
        raw.append(field);assert io.sha(io.RUNS/'raw'/f'{i:03d}.npz')==a['raw_sha256'][str(i)]
        unique={}
        for q in sorted(anchors[i],key=lambda q:q['id']):
            key=tuple((np.array(q['uv'])//p['calibration']['deduplicate_px']).astype(int))
            if key not in unique:unique[key]=q
        rows=list(unique.values());assert [q['id'] for q in rows]==c['anchor_ids']
        assert np.allclose([q['depth'] for q in rows],c['depth'],atol=1e-12,rtol=1e-12)
        if c['coefficients'] is not None:
            uv=np.array(c['uv']);z=np.array(c['depth']);x=samples(field,uv);ids=np.array(c['fit_ids'])
            expected=affine(x[ids],1/z[ids]);assert np.allclose(expected,c['coefficients'],atol=1e-12,rtol=1e-12)
            coefficients_checked+=1;held=np.array(c['check_ids']);inv=expected[0]*x[held]+expected[1]
            if 'check_median_relative' in c:
                error=abs(1/inv-z[held])/z[held]
                assert np.isclose(np.median(error),c['check_median_relative'],atol=1e-12)
                assert np.isclose(np.quantile(error,.9),c['check_P90_relative'],atol=1e-12)
            if 'split_coefficients' in c:
                s=p['calibration'];ok=(c['check_median_relative']<=s['relative_median_max'] and c['check_P90_relative']<=s['relative_P90_max']
                    and c['split_median_relative']<=s['relative_median_max'] and c['split_P90_relative']<=s['relative_P90_max'])
                assert c['passed']==ok
    for q in a['motion']:
        if 'row' not in q:continue
        start=q['start'];uv=np.array(q['uv']);z=np.array(q['prior_depth'],float)
        rays=np.linalg.solve(k,np.c_[uv,np.ones(12)].T).T
        world=[]
        for t in range(12):
            view=views[start+t];world.append(view[:3,:3].T@(rays[t]*z[t]-view[:3,3]))
        assert np.allclose(world,np.array(q['prior_world'],float),atol=1e-10,rtol=1e-10,equal_nan=True)
        assert np.array_equal(xyz[start:start+12,q['row']],q['parent_world'])
        valid=np.array(q['local_eligible']);fit=np.flatnonzero(valid&(np.arange(12)%2==0));held=np.flatnonzero(valid&(np.arange(12)%2==1))
        assert q['valid']==int(valid.sum()) and q['fit_count']==len(fit) and q['held_count']==len(held)
        if len(fit)<4 or len(held)<4:assert q['reason']=='insufficient_observations' and not q['passed']
        motion_checked+=1
    groups={key:dict(observations=0,learned_visibility_failure=0,bidirectional_disagreement=0,track_rejected=0)
            for key in ['boundary_proxy','interior_proxy','unknown']}
    offsets=np.array([(x,y) for y in range(-2,3) for x in range(-2,3)])
    for start in range(0,109,4):
        f=np.load(io.ROOT/f'experiments/staged_reconstruction/RubberDuck_I10_tracking/dense/v1/clip_{start:03d}.npz')
        for t in range(12):
            i=start+t;points=f['forward'][t];labels=np.full(len(points),'unknown',object);c=a['calibration'][i]
            good=np.isfinite(points).all(1)&(points[:,0]>=2)&(points[:,0]<=397)&(points[:,1]>=2)&(points[:,1]<=222)
            if c['passed']:
                ids=np.flatnonzero(good);uv=(points[ids,None]+offsets).reshape(-1,2)
                inverse=c['coefficients'][0]*samples(raw[i],uv)+c['coefficients'][1]
                depth=np.full(len(inverse),np.nan);pos=np.isfinite(inverse)&(inverse>0);depth[pos]=1/inverse[pos]
                depth=depth.reshape(len(ids),25);valid=np.isfinite(depth).all(1)
                contrast=(depth.max(1)-depth.min(1))/np.median(depth,axis=1)
                labels[ids[valid]]='interior_proxy';labels[ids[valid&(contrast>.1)]]='boundary_proxy'
            for key in groups:
                select=labels==key;g=groups[key];g['observations']+=int(select.sum())
                g['learned_visibility_failure']+=int((select&~(f['visible'][t]&f['reverse_visible'][t])).sum())
                g['bidirectional_disagreement']+=int((select&(np.linalg.norm(f['forward'][t]-f['reverse'][t],axis=1)>.75)).sum())
                g['track_rejected']+=int((select&~f['accepted']).sum())
    assert groups==a['boundary_groups']
    s=a['summary'];assert sum(q['observations'] for q in groups.values())==s['boundary_observation_denominator']==53196
    assert s['calibrated_views']==sum(q['passed'] for q in a['calibration'])
    assert s['reliable_motion_tracks']==sum(q['passed'] for q in a['motion'])
    assert s['auxiliary_observations']==sum(len(q) for q in a['eligible_fit_observations'].values())
    io.verify();io.write(io.REPORT/'independent_audit.json',dict(passed=True,
        all120_RGB_stationary_anchor_deduplications=True,independent_affine_checks=coefficients_checked,
        all120_raw_SHA=True,all_motion_world_backprojections=motion_checked,
        all53196_boundary_observations_exact=True,all_counts_and_zero_targets_checked=True,
        source_input_parent_unchanged=True,optimizer_updates=0,source_sha256=io.sha(Path(__file__))))
    print('L7_INDEPENDENT_ARITHMETIC_AND_DENOMINATOR_PASS',coefficients_checked,motion_checked,flush=True)


if __name__=='__main__':main()
