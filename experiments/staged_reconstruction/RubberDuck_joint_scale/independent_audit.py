"""Separate arithmetic for grouping, robust objective, held checks and motion."""
import argparse
import hashlib
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import Delaunay,ConvexHull,QhullError
from variants.staged_reconstruction.i10_joint_scale import io


def sample(raw,uv):
    uv=np.asarray(uv,np.float32).reshape(-1,2)
    return cv2.remap(raw.astype(np.float32),uv[:,0:1],uv[:,1:2],cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT).reshape(-1).astype(float)


def z_at(raw,uv,c):
    inv=sample(raw,uv)*c[0]+c[1]
    return np.divide(1,inv,out=np.full(len(inv),np.nan),where=np.isfinite(inv)&(inv>0))


def hull(uv,anchors):
    try:return Delaunay(np.asarray(anchors)).find_simplex(uv)>=0
    except QhullError:return np.zeros(len(uv),bool)


def jumps(raw,uv,c):
    uv=np.asarray(uv).reshape(-1,2)
    offsets=np.array([(x,y) for y in range(-2,3) for x in range(-2,3)])
    positions=uv[:,None,:]+offsets[None]
    inside=((positions[:,:,0]>=0)&(positions[:,:,0]<=399)&(positions[:,:,1]>=0)&(positions[:,:,1]<=224)).all(1)
    z=z_at(raw,positions.reshape(-1,2),c).reshape(-1,25)
    valid=inside&np.isfinite(z).all(1);contrast=np.full(len(uv),np.nan)
    contrast[valid]=np.ptp(z[valid],axis=1)/np.median(z[valid],axis=1)
    return contrast,valid


def close(a,b):
    assert np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True),(a,b)


def run():
    cv2.setNumThreads(1);io.verify()
    for receipt in ['extraction_freeze.json','numerical_repair_freeze.json']:
        for path,h in io.read(io.REPORT/receipt)['files'].items():assert io.sha(io.ROOT/path)==h,path
    a=io.read(io.REPORT/'audit_v2.json');p=io.read(io.BASE/'protocol.json')
    op=io.read(io.ROOT/p['old_protocol']);tp=io.read(io.ROOT/p['tracking_protocol'])
    asset=np.load(io.ROOT/op['trajectories']);views,k=asset['views'].astype(float),asset['k'].astype(float)
    rows=[dict(q,id='old_'+q['id']) for q in io.read(io.ROOT/op['tracks'])['rows'] if q['stationarity']['label']=='stationary_compatible']
    newrows=[]
    for start in p['starts']:
        q=io.read(io.RUNS/'clips'/f'{start:03d}.json')
        newrows.extend(r for r in q['rows'] if r['stationarity']['label']=='stationary_compatible')
    rows=sorted(rows+newrows,key=lambda q:q['id']);parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i,q in enumerate(rows):
        for j in range(i):
            r=rows[j];start=max(q['start'],r['start']);end=min(q['start']+12,r['start']+12)
            if end-start<2:continue
            x=np.array(q['uv'])[start-q['start']:end-q['start']];y=np.array(r['uv'])[start-r['start']:end-r['start']]
            if np.max(np.linalg.norm(x-y,axis=1))<=1:
                x,y=root(i),root(j);parent[max(x,y)]=min(x,y)
    groups={q['id']:rows[root(i)]['id'] for i,q in enumerate(rows)}
    assert groups==a['grouping']['groups']
    bank=[[] for _ in range(120)]
    for q in rows:
        group=groups[q['id']];n=int(hashlib.sha256(group.encode()).hexdigest()[:16],16)
        split='check' if n%3==0 else ('fit0' if (n//3)%2==0 else 'fit1')
        point=np.array(q['stationarity']['stationary_hypothesis_xyz'])
        for t,uv in enumerate(q['uv']):
            i=q['start']+t;camera=views[i,:3,:3]@point+views[i,:3,3];pixel=k@camera;pixel=pixel[:2]/pixel[2]
            assert camera[2]>tp['minimum_positive_depth']
            assert np.linalg.norm(pixel-uv)<=tp['stationary_max_error_px']+1e-10
            bank[i].append(dict(id=q['id'],group=group,split=split,uv=uv,depth=float(camera[2])))
    for i,values in enumerate(bank):
        cells=defaultdict(list)
        for q in values:cells[tuple((np.array(q['uv'])//4).astype(int))].append(q)
        chosen=[]
        for _,qs in sorted(cells.items()):
            held=[q for q in qs if q['split']=='check'];chosen.append(sorted(held or qs,key=lambda q:q['id'])[0])
        bank[i]=chosen
        saved=a['anchor_bank'][i];assert len(saved)==len(chosen)
        for q,r in zip(chosen,saved):
            assert all(q[k]==r[k] for k in ['id','group','split','uv']);close(q['depth'],r['depth'])
    raw=[]
    for i in range(120):
        with np.load(io.ROOT/p['raw_maps']/f'{i:03d}.npz') as f:raw.append(f['relative'].copy())
    objectives=[];gradient_checks=[]
    for solution,half in zip(a['joint_solutions'],[None,0,1]):
        assert solution['success'];theta=np.array(solution['normalized_coefficients']);gradient=np.zeros_like(theta);value=0.
        for i,anchors in enumerate(bank):
            qs=[q for q in anchors if q['split']!='check' and (half is None or q['split']==f'fit{half}')]
            if not qs:continue
            med,spread=float(np.median(raw[i])),float(np.quantile(raw[i],.9)-np.quantile(raw[i],.1))
            x=((sample(raw[i],[q['uv'] for q in qs]).astype(np.float32)-med)/spread).astype(float)
            y=1/np.array([q['depth'] for q in qs])
            r=(theta[i,0]*x+theta[i,1])/y-1;d=p['joint']['huber_scale']
            value+=np.mean(np.where(np.abs(r)<=d,.5*r*r,d*(np.abs(r)-.5*d)))
            g=np.clip(r,-d,d)/y/len(qs);gradient[i]+=[np.sum(g*x),np.sum(g)]
            close(solution['coefficients'][i],[theta[i,0]/spread,theta[i,1]-theta[i,0]*med/spread])
        second=np.diff(theta,n=2,axis=0);factor=p['joint']['smoothness']/solution['scale']**2
        value+=.5*factor*np.sum(second**2)+.5*p['joint']['ridge']/solution['scale']**2*np.sum(theta**2)
        gradient[:-2]+=factor*second;gradient[1:-1]-=2*factor*second;gradient[2:]+=factor*second
        gradient+=p['joint']['ridge']/solution['scale']**2*theta
        projected=gradient.copy();bound=theta[:,0]<=p['joint']['minimum_slope']+1e-10
        projected[bound,0]=np.minimum(projected[bound,0],0)
        # Independent numerical-solver check, never a geometry/admission gate.
        assert abs(value-solution['cost'])<1e-10
        assert np.max(np.abs(projected))<1e-7
        objectives.append(value);gradient_checks.append(float(np.max(np.abs(projected))))
    held_metrics=0
    for name in ['candidate','framewise_control']:
        case=a[name];cal=case['calibration']
        for i,c in enumerate(cal):
            if 'check_P90_relative' not in c:continue
            qs=[q for q in bank[i] if q['split']=='check'];uv=[q['uv'] for q in qs];z=np.array([q['depth'] for q in qs])
            predicted=z_at(raw[i],uv,c['coefficients']);s0,s1=[z_at(raw[i],uv,v) for v in c['split_coefficients']]
            error=np.abs(predicted-z)/z;stability=np.abs(s0-s1)/z
            close(c['check_median_relative'],np.median(error));close(c['check_P90_relative'],np.quantile(error,.9))
            close(c['split_median_relative'],np.median(stability));close(c['split_P90_relative'],np.quantile(stability,.9))
            assert c['passed']==bool(np.median(error)<=.05 and np.quantile(error,.9)<=.1 and np.median(stability)<=.05 and np.quantile(stability,.9)<=.1)
            held_metrics+=1
        motion_count=0;expected_targets={}
        old={q['id']:q for q in io.read(io.ROOT/p['old_audit'])['motion']}
        for q in case['motion']:
            if 'row' not in q:continue
            motion_count+=1;start=q['start'];uv=np.array(q['uv']);z=np.full(12,np.nan);valid=np.zeros(12,bool);reasons=[]
            for t in range(12):
                i=start+t;c=cal[i];reason='uncalibrated'
                if c['passed']:
                    inside=hull(uv[t:t+1],np.array(c['uv'])[c['fit_ids']])[0]
                    jump,good=jumps(raw[i],uv[t:t+1],c['coefficients']);z[t]=z_at(raw[i],uv[t:t+1],c['coefficients'])[0]
                    valid[t]=inside and good[0] and jump[0]<=.1 and np.isfinite(z[t])
                    reason='passed_local_interpolation' if valid[t] else ('outside_fit_hull' if not inside else 'boundary_or_invalid')
                reasons.append(reason)
            assert valid.tolist()==q['local_eligible'] and reasons==q['local_reasons'];close(z,np.array(q['prior_depth'],float))
            fit=np.flatnonzero(valid&(np.arange(12)%2==0));held=np.flatnonzero(valid&(np.arange(12)%2==1))
            assert q['fit_count']==len(fit) and q['held_count']==len(held)
            if len(fit)<4 or len(held)<4:assert not q['passed'];continue
            world=[]
            for t in range(12):
                ray=np.linalg.solve(k,np.r_[uv[t],1])*z[t]
                world.append(np.linalg.solve(views[start+t,:3,:3],ray-views[start+t,:3,3]))
            world=np.array(world);offset=world-np.array(old[q['id']]['parent_world']);reference=np.median(offset[fit],axis=0)
            errors=np.linalg.norm(offset[held]-reference,axis=1)/z[held]
            close(reference,q['fit_offset']);close(np.median(errors),q['motion_median_relative']);close(np.quantile(errors,.9),q['motion_P90_relative'])
            assert q['passed']==bool(np.median(errors)<=.02 and np.quantile(errors,.9)<=.05)
            if q['passed']:
                for t in fit:
                    i=start+int(t);c=cal[i];confidence=max(0,1-c['check_P90_relative']/.1)*max(0,1-q['motion_P90_relative']/.05)
                    expected_targets.setdefault(str(i),[]).append(dict(id=q['id'],row=q['row'],uv=uv[t].tolist(),depth=z[t],confidence=confidence))
        assert motion_count==122
        assert expected_targets==case['targets']
        boundary={k:dict(observations=0,learned_visibility_failure=0,bidirectional_disagreement=0) for k in ['boundary_proxy','interior_proxy','unknown']}
        for start in p['starts']:
            with np.load(io.ROOT/f'experiments/staged_reconstruction/RubberDuck_I10_tracking/dense/v1/clip_{start:03d}.npz') as f:
                for t in range(12):
                    uv=f['forward'][t];c=cal[start+t];labels=np.full(len(uv),'unknown',object)
                    ids=np.flatnonzero(np.isfinite(uv).all(1)&(uv[:,0]>=2)&(uv[:,0]<=397)&(uv[:,1]>=2)&(uv[:,1]<=222))
                    if c['passed']:
                        jump,good=jumps(raw[start+t],uv[ids],c['coefficients'])
                        labels[ids[good]]='interior_proxy';labels[ids[good&(jump>.1)]]='boundary_proxy'
                    missing=~(f['visible'][t].astype(bool)&f['reverse_visible'][t].astype(bool));diff=np.linalg.norm(f['forward'][t]-f['reverse'][t],axis=1)>.75
                    for key,g in boundary.items():
                        selected=labels==key;g['observations']+=int(selected.sum());g['learned_visibility_failure']+=int((selected&missing).sum());g['bidirectional_disagreement']+=int((selected&diff).sum())
        assert boundary==case['boundary_groups'] and sum(g['observations'] for g in boundary.values())==53196
    io.verify()
    io.write(io.REPORT/'independent_audit.json',dict(passed=True,stationary_tracks_checked=len(rows),expanded_stationary_reprojections=len(newrows)*12,
        independent_duplicate_groups=len(set(groups.values())),all120_anchor_banks_checked=True,robust_objectives=objectives,
        independent_projected_gradients=gradient_checks,held_frame_checks=held_metrics,motion_rows_per_control=122,
        boundary_observations_per_control=53196,all120_raw_and_all_frozen_SHA=True,
        optimizer_updates=0,independent_geometry_GT=False,source_sha256=io.sha(Path(__file__))))
    print('JOINT_SCALE_INDEPENDENT_ARITHMETIC_PASS',len(rows),held_metrics,gradient_checks,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-joint-scale-audit',action='store_true');args=p.parse_args()
    if not args.enable_joint_scale_audit:raise SystemExit('Joint scale audit default off')
    with io.lock():run()


if __name__=='__main__':main()
