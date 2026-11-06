"""Independent pair arithmetic and saved native controller replay; no updates."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import torch
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

B=ROOT/'experiments/staged_reconstruction/RubberDuck_relative_order'
R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_relative_order/v1'


def pair_arithmetic():
    p=read(B/'protocol.json');e=read(ROOT/p['evidence']);bank=read(B/'v1/pair_bank.json')
    raw=[]
    for i in range(120):
        with np.load(ROOT/f'runs/pidg_staged_reconstruction/RubberDuck_pseudo_depth/v1/raw/{i:03d}.npz') as f:raw.append(f['relative'].copy())
        scale=float(np.quantile(raw[-1],.9)-np.quantile(raw[-1],.1))
        assert abs(scale-e['normalizations'][i][1])<1e-12
    def interval(i,point):
        image=raw[i];u,v=np.asarray(point,float);h,w=image.shape;scale=e['normalizations'][i][1]
        if not np.isfinite([u,v]).all() or u<2 or v<2 or u>w-3 or v>h-3 or scale<=1e-12:return None
        offsets=np.mgrid[-2:3,-2:3]
        # Independent coordinate construction; same declared OpenCV interpolation.
        x=(u+offsets[1]).astype(np.float32);y=(v+offsets[0]).astype(np.float32)
        z=cv2.remap(image.astype(np.float32),x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(float)
        if not np.isfinite(z).all() or (z.max()-z.min())/scale>p['maximum_raw_patch_range']:return None
        return z.min(),z.max()
    results={};group_sets={}
    for split in ['fit','held']:
        observed=bank[split];hypotheses=[];proposals=[];groups_seen=set()
        for track in e['candidate']['motion']:
            if 'row' not in track:continue
            groups={};start=track['start']
            for t in range(12):
                for q in e['anchor_bank'][start+t]:
                    if (q['split']=='check')!=(split=='held'):continue
                    groups.setdefault(q['group'],{}).setdefault(t,q)
            groups_seen.update(groups)
            for group,anchors in sorted(groups.items()):
                signs=[0]*12;gaps=[None]*12
                for t,anchor in anchors.items():
                    i=start+t;u=np.asarray(track['uv'][t]);v=np.asarray(anchor['uv'])
                    if np.linalg.norm(u-v)<p['minimum_pixel_separation']:continue
                    x=interval(i,u);y=interval(i,v);scale=e['normalizations'][i][1]
                    if x is None or y is None:continue
                    if x[0]>y[1]+p['raw_normalized_gap_min']*scale:signs[t]=1;gaps[t]=(x[0]-y[1])/scale
                    elif y[0]>x[1]+p['raw_normalized_gap_min']*scale:signs[t]=-1;gaps[t]=(y[0]-x[1])/scale
                fit=[t for t in range(0,12,2) if signs[t]];check=[t for t in range(1,12,2) if signs[t]]
                sign=0;reason='passed'
                if len(fit)<p['minimum_order_fit_observations']:reason='insufficient_fit_orders'
                elif any(signs[t]!=signs[fit[0]] for t in fit):reason='fit_order_conflict'
                else:
                    sign=signs[fit[0]]
                    if len(check)<p['minimum_order_check_observations']:reason='insufficient_check_orders'
                    elif any(signs[t]!=sign for t in check):reason='held_order_conflict'
                hypotheses.append(dict(track=track['id'],row=track['row'],start=start,anchor_group=group,
                    signs=signs,raw_gaps=gaps,fit_count=len(fit),check_count=len(check),fit_chosen_sign=sign,reason=reason))
                if reason!='passed':continue
                for t in (check if split=='held' else fit):
                    q=anchors[t];i=start+t
                    proposals.append(dict(id=f'{track["id"]}:{i}:{group}',track=track['id'],row=track['row'],frame=i,
                        anchor=q['id'],anchor_group=group,uv=[track['uv'][t],q['uv']],sign=sign,
                        confidence=.5,auxiliary_held=split=='held',reason='passed'))
        counts=Counter();chosen=[]
        for q in sorted(proposals,key=lambda q:(q['track'],q['anchor_group'],q['frame'])):
            if counts[q['frame']]>=p['maximum_pairs_per_frame']:q['reason']='fixed_pair_cap'
            else:chosen.append(q.copy());counts[q['frame']]+=1
        assert hypotheses==observed['hypotheses'],split
        assert proposals==observed['proposals'] and chosen==observed['pairs'],split
        group_sets[split]=groups_seen
        results[split]=dict(all_hypotheses=len(hypotheses),all_proposals=len(proposals),pairs=len(chosen),
            hypothesis_reasons=dict(Counter(q['reason'] for q in hypotheses)),selected_per_frame=dict(counts),
            dynamic_rows=sorted({q['row'] for q in chosen}))
    assert not group_sets['fit']&group_sets['held']
    return dict(passed=True,all120_raw_normalizations=True,independent_patch_interval_signs=True,
        even_fit_selects_odd_only_rejects=True,whole_anchor_group_split_disjoint=True,stable_global_cap_exact=True,
        splits=results,independent_geometry_GT=False)


def next_state(previous,m):
    out={k:deepcopy(v) for k,v in previous.items() if k!='measurement'}
    if not m['known']:out.update(status='suspended',passes=0)
    elif m['passed']:
        out.update(passes=out['passes']+1,failures=0)
        out['status']='active' if out['passes']>=2 else 'suspended'
    else:
        out.update(failures=out['failures']+1,passes=0)
        out['status']='revoked' if out['failures']>=2 else 'suspended'
    return dict(**out,measurement=m)


def controller_replay():
    from variants.staged_reconstruction.i10_relative_order import runtime as a
    r=a.install(True);r.verify();assert not r.stopped();torch.set_num_threads(4)
    initial=read(B/'v1/initial_health.json')['controller'];reports={}
    for arm in ['rgb_control','ordinal_depth']:
        renderer,scene,opts,_,_=r.load(arm);previous=deepcopy(initial);rows=[]
        for step in [100,200,300,400,500]:
            path=R/'refresh'/arm/f'model_{step:04d}.pt';receipt=read(path.with_suffix('.json'))
            assert sha(path)==receipt['model_sha256'];saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['model_only_not_resume'];scene.load_state_dict(saved['model'],strict=True)
            before=receipt['before'];after=receipt['after']
            assert before['calls']==step and before['pairs']==previous['pairs'] and before['history']==previous['history']
            measured=a.measure(renderer,scene)
            expected={k:next_state(before['pairs'][k],m) for k,m in measured.items()}
            assert expected==after['pairs'],(arm,step)
            assert after['calls']==step and after['history'][:-1]==before['history']
            summary=after['history'][-1]
            assert summary==dict(completed_updates=step,pairs=30,
                **{s:sum(v['status']==s for v in expected.values()) for s in ['active','suspended','revoked']})
            rows.append(summary);previous=after;print('L8R_CONTROLLER_REPLAY',arm,step,summary,flush=True)
        final=torch.load(R/'formal'/arm/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert final['controller']['calls']==600 and final['controller']['pairs']==previous['pairs']
        assert final['controller']['history']==previous['history']
        usage=read(R/'formal'/arm/'usage.json');assert len(usage)==600
        assert [q['index'] for q in usage]==final['sample_trace']
        expected_pairs={i:sum(q['frame']==i for q in a.BANK['fit']['pairs']) for i in range(120)}
        for q in usage:
            assert 0<=q['violating_pairs']<=q['used_pairs']<=q['total_pairs']
            assert q['total_pairs']==(expected_pairs[q['index']] if arm=='ordinal_depth' else 0)
        positive=[q for q in usage if q['weighted_ordinal_loss']>0]
        if arm=='ordinal_depth':assert positive,'No actual auxiliary updates'
        else:assert not positive
        reports[arm]=dict(refreshes=rows,usage_steps=600,used_pair_observations=sum(q['used_pairs'] for q in usage),
            steps_using_pairs=sum(q['used_pairs']>0 for q in usage),positive_auxiliary_steps=len(positive),
            sum_weighted_auxiliary_loss=sum(q['weighted_ordinal_loss'] for q in usage))
    r.verify();return dict(passed=True,ten_native_model_replays_exact=True,independent_state_transitions_exact=True,
        all1200_usage_rows_checked=True,arms=reports,optimizer_updates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--enable-evidence-audit',action='store_true');p.add_argument('--phase',choices=['pairs','controller'],required=True);args=p.parse_args()
    if not args.enable_evidence_audit:raise SystemExit('Read-only audit default off')
    output=B/f'v1/{args.phase}_audit.json';assert not output.exists()
    value=pair_arithmetic() if args.phase=='pairs' else controller_replay()
    write(output,dict(**value,source_sha256=sha(Path(__file__))));print('L8R_EVIDENCE_AUDIT_PASS',args.phase,flush=True)
