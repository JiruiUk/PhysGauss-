"""Bounded controlled and natural tests; no Gaussian training or ID fitting."""
from pathlib import Path
import sys,json,hashlib,time
from dataclasses import asdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.motion_correspondence.model import Config,Observation,MotionTracker,predict
REPORT=ROOT/'experiments/motion_correspondence';OUT=ROOT/'output/motion_correspondence'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
VARIANTS={'no_history':Config(use_history=False,uncertainty_guard=False),
          'history':Config(use_history=True,uncertainty_guard=False),
          'history_guarded':Config(use_history=True,uncertainty_guard=True)}


def observations(component):
    result=[]
    for k in range(int(component.max()+1)):
        mask=component==k;y,x=np.where(mask)
        result.append(Observation(np.array([x.mean(),y.mean()]),float(len(x)),k,mask))
    return result


def synthetic():
    output={}
    for scene in ['hidden_pass','hidden_bounce','symmetric_return']:
        output[scene]={}
        for name,c in VARIANTS.items():
            initial=[Observation(np.array([10.,20.]),25.,0),Observation(np.array([50.,20.]),25.,1)]
            tr=MotionTracker(initial,0.,c)
            for n in range(1,6):tr.step([Observation(np.array([10.+2*n,20.]),25.,0),Observation(np.array([50.-2*n,20.]),25.,1)],n/199)
            for n in range(6,12):tr.step([],n/199)
            if scene=='symmetric_return':
                # A genuinely unidentifiable observation geometry, not a
                # declared crossing or a made-up ground truth identity.
                target=[Observation(np.array([30.,16.]),25.,0),Observation(np.array([30.,24.]),25.,1)]
                truth=None
            else:
                target=[Observation(np.array([26.,20.]),25.,0),Observation(np.array([34.,20.]),25.,1)]
                truth=[1,0] if scene=='hidden_pass' else [0,1]
            before=np.array([predict(s,12/199-tr.time,c).mean[:2] for s in tr.hypotheses[0]['states']])
            r=tr.step(target,12/199)
            output[scene][name]={'accepted':r['accepted_assignment'],'best':r['best_assignment'],'margins':r['assignment_margins'],
              'alternatives':r['alternatives'],'reference_assignment':truth,'forecast_before_target_observation':before.tolist(),
              'accepted_correct':None if truth is None else sum(j>=0 and j==truth[i] for i,j in enumerate(r['accepted_assignment'])),
              'accepted_wrong':None if truth is None else sum(j>=0 and j!=truth[i] for i,j in enumerate(r['accepted_assignment'])),
              'abstained':sum(j<0 for j in r['accepted_assignment'])}
            if scene!='symmetric_return':
                sequel=[]
                for n in range(13,21):
                    more=[Observation(np.array([50.-2*n,20.]),25.,0),Observation(np.array([10.+2*n,20.]),25.,1)]
                    sr=tr.step(more,n/199)
                    sequel.append({'frame':n,'accepted':sr['accepted_assignment'],
                        'unresolved_seed_identities':sr['unresolved_seed_identities'],
                        'accepted_wrong':sum(j>=0 and j!=truth[i] for i,j in enumerate(sr['accepted_assignment']))})
                output[scene][name]['identical_observation_sequel']=sequel
    write(REPORT/'synthetic.json',output);return output


def qualified_reference(component,labels):
    mapping={}
    for o in observations(component):
        known=labels[o.mask];known=known[known>=0]
        if len(known)<30:return None
        count=np.bincount(known,minlength=4);winner=int(count.argmax())
        if count[winner]/len(known)<.95:return None
        mapping[o.component]=winner
    if set(mapping.values())!={0,1,2,3}:return None
    return mapping


def controlled(frames,components,reference_labels):
    plan=read(REPORT/'plan.json')['real_controlled_test'];prefix=np.where(frames<=42)[0]
    assert len(prefix)==25
    references=[qualified_reference(components[i],reference_labels[i]) for i in prefix]
    assert all(r is not None for r in references),'Pseudo-reference qualification failed; do not invent labels'
    trials=[]
    for start in plan['gap_start_indices']:
        for gap in plan['missing_training_observations']:
            target=start+gap
            for hidden in plan['hidden_seed_ID_sets']:
                trial={'start_frame':int(frames[start]),'return_frame':int(frames[target]),'missing_observations':gap,'hidden_seed_IDs':hidden,'variants':{}}
                for name,c in VARIANTS.items():
                    tr=MotionTracker(observations(components[0]),(int(frames[0])-1)/199,c);target_result=None
                    for i in range(1,target+1):
                        visible=observations(components[i])
                        if start<=i<target:visible=[o for o in visible if references[i][o.component] not in hidden]
                        t=(int(frames[i])-1)/199
                        if i==target:
                            forecast=np.array([predict(s,t-tr.time,c).mean[:2] for s in tr.hypotheses[0]['states']])
                        result=tr.step(visible,t)
                        if i==target:target_result=result
                    reference_xy={references[target][o.component]:o.xy for o in visible}
                    errors=[float(np.linalg.norm(forecast[k]-reference_xy[k])) for k in hidden]
                    outcomes=[]
                    for identity in hidden:
                        j=target_result['accepted_assignment'][identity]
                        assigned=None if j<0 else references[target][visible[j].component]
                        outcomes.append({'seed_ID':identity,'assigned_pseudo_reference_ID':assigned,
                            'correct':None if assigned is None else assigned==identity,
                            'cost_margin':target_result['assignment_margins'][identity]})
                    trial['variants'][name]={'forecast_pixel_errors':errors,'outcomes':outcomes,'best_assignment':target_result['best_assignment']}
                trials.append(trial)
                print('controlled',len(trials),trial['start_frame'],trial['return_frame'],hidden,flush=True)
    summary={}
    for name in VARIANTS:
        errors=[e for r in trials for e in r['variants'][name]['forecast_pixel_errors']]
        outcomes=[o for r in trials for o in r['variants'][name]['outcomes']]
        accepted=[o for o in outcomes if o['correct'] is not None]
        summary[name]={'forecast_error_mean_pixels':float(np.mean(errors)),'forecast_error_median_pixels':float(np.median(errors)),
          'cases':len(outcomes),'accepted':len(accepted),'correct':sum(o['correct'] for o in accepted),'wrong':sum(not o['correct'] for o in accepted),
          'abstained':len(outcomes)-len(accepted),'accepted_accuracy':sum(o['correct'] for o in accepted)/len(accepted) if accepted else None}
    report={'reference_is_pseudo_not_material_GT':True,'hidden_observations_not_supplied_to_tracker':True,
      'forecast_computed_before_reappearance_observation':True,'qualified_reference_frames':len(prefix),'trials':trials,'summary':summary}
    write(REPORT/'controlled.json',report);print('CONTROLLED',json.dumps(summary),flush=True);return report


def natural(frames,components):
    output={}
    for name,c in VARIANTS.items():
        initial=observations(components[0]);assert len(initial)==4
        tr=MotionTracker(initial,(int(frames[0])-1)/199,c)
        rows=[{'frame':int(frames[0]),'accepted_assignment':list(range(4)),'best_assignment':list(range(4)),
               'assignment_margins':[None]*4,'forecast':[o.xy.tolist() for o in initial],'merged_components':[],'retained_hypotheses':1}]
        label_maps=[components[0].copy()]
        for number,component in zip(frames[1:],components[1:]):
            obs=observations(component);r=tr.step(obs,(int(number)-1)/199);r['frame']=int(number)
            canvas=np.full(component.shape,-1,dtype=np.int8)
            for identity,j in enumerate(r['accepted_assignment']):
                if j>=0:canvas[component==obs[j].component]=identity
            rows.append(r);label_maps.append(canvas)
            if len(rows)%20==0:print('natural',name,int(number),r['accepted_assignment'],flush=True)
        np.savez_compressed(OUT/(name+'.npz'),frames=frames,labels=np.array(label_maps,dtype=np.int8))
        report={'config':asdict(c),'note':'Candidate image correspondences only; natural ID correctness is unverified. Beam score margins are not calibrated probabilities.',
          'frames':rows,'accepted_frame_ID_events':sum(j>=0 for r in rows for j in r['accepted_assignment']),
          'total_frame_ID_events':len(rows)*4,'no_Gaussian_ID_reassignment':True}
        write(REPORT/(name+'.json'),report);output[name]=report
    return output


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    files=[REPORT/'plan.json',ROOT/'variants/motion_correspondence/model.py',Path(__file__),ROOT/'tests/test_motion_correspondence.py',
       ROOT/'output/surface_support/image_tracks.npz',ROOT/'experiments/surface_support/input_lock.json',ROOT/'experiments/surface_support/support_gate.json']
    old=read(ROOT/'experiments/surface_support/input_lock.json')
    before={str(p.relative_to(ROOT)):sha(p) for p in files};before.update({p:v for p,v in old.items() if not p.startswith('output/surface_support/') and not p.startswith('tests/')})
    assert all(sha(ROOT/p)==v for p,v in before.items());write(REPORT/'input_lock.json',before)
    start=time.time();z=np.load(ROOT/'output/surface_support/image_tracks.npz');frames=z['frames'];components=z['components'];reference=z['labels']
    assert len(frames)==120
    sy=synthetic();co=controlled(frames,components,reference);na=natural(frames,components)
    assert all(sha(ROOT/p)==v for p,v in before.items())
    write(REPORT/'verification.json',{'passed':True,'input_model_graph_data_source_SHA_unchanged':True,'training_optimizer_steps':0,
        'new_model_type':'Non-neural multi-hypothesis image-plane motion association', 'elapsed_seconds':time.time()-start,
        'synthetic_tests':'synthetic.json','controlled_trials':len(co['trials']),'natural_training_frames_each':len(frames),
        'no_material_identity_GT_claim':True,'manual_review_pending':True})
    print('BOUNDED MOTION CORRESPONDENCE TEST COMPLETE',flush=True)


if __name__=='__main__':main()
