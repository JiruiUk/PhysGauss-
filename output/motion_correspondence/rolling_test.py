"""Local rolling pseudo-occlusion diagnostic; no global identities are made."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.motion_correspondence.run import REPORT,OUT,read,write,sha,VARIANTS,observations
from variants.motion_correspondence.model import MotionTracker,predict
from output.surface_support.core import propagate


def mapping(component,labels):
    if int(component.max())!=3:return None,'component_count_not_four'
    values={}
    for obs in observations(component):
        known=labels[obs.mask];known=known[known>=0]
        if len(known)<30:return None,'insufficient_visible_FB_support'
        counts=np.bincount(known,minlength=4);winner=int(counts.argmax())
        if counts[winner]/len(known)<.95:return None,'mixed_visible_FB_labels'
        values[obs.component]=winner
    if set(values.values())!={0,1,2,3}:return None,'non_bijective_visible_reference'
    return values,None


def main():
    torch.set_num_threads(4)
    lock=read(REPORT/'input_lock.json');assert all(sha(ROOT/p)==v for p,v in lock.items())
    extra=[Path(__file__),REPORT/'rolling_plan.json',ROOT/'output/surface_support/core.py']
    extra_lock={str(p.relative_to(ROOT)):sha(p) for p in extra};write(REPORT/'rolling_input_lock.json',extra_lock)
    z=np.load(ROOT/'output/surface_support/image_tracks.npz');frames=z['frames'];components=z['components']
    flow_dir=ROOT/'data/derived/flow_comparison_train_only';manifest=read(flow_dir/'manifest.json')
    pairs={}
    for row in manifest['pairs']:
        with np.load(flow_dir/row['file']) as f:
            pairs[(row['source'],row['target'])]=(torch.from_numpy(f['forward'].astype(np.float32)),torch.from_numpy(f['backward'].astype(np.float32)))
    trials=[];excluded=[]
    for start in range(0,109,6):
        for gap in [2,4]:
            target=start+5+gap;labels=components[start].copy();refs=[];reason=None
            for index in range(start,target+1):
                if index>start:
                    f,b=pairs[(int(frames[index-1]),int(frames[index]))]
                    labels,ambiguous=propagate(labels,components[index],f,b)
                    if ambiguous:reason='visible_FB_merge_ambiguity';break
                value,reason=mapping(components[index],labels)
                if reason:break
                refs.append(value)
            row={'start_index':start,'start_frame':int(frames[start]),'return_frame':int(frames[target]),'hidden_observations':gap,
                 'stratum':'early' if frames[start]<60 else ('middle' if frames[start]<140 else 'late')}
            if reason:
                row.update({'reason':reason,'rejected_at_frame':int(frames[index])});excluded.append(row);continue
            row['variants']={}
            for name,config in VARIANTS.items():
                tracker=MotionTracker(observations(components[start]),(int(frames[start])-1)/199,config)
                for index in range(start+1,target):
                    obs=observations(components[index]) if index<start+5 else []
                    tracker.step(obs,(int(frames[index])-1)/199)
                t=(int(frames[target])-1)/199
                forecast=np.array([predict(s,t-tracker.time,config).mean[:2] for s in tracker.hypotheses[0]['states']])
                obs=observations(components[target]);reference={refs[-1][o.component]:o.xy for o in obs}
                result=tracker.step(obs,t);outcomes=[]
                for identity,j in enumerate(result['accepted_assignment']):
                    assigned=None if j<0 else refs[-1][obs[j].component]
                    outcomes.append({'local_seed':identity,'assigned_local_reference':assigned,'correct':None if assigned is None else assigned==identity})
                row['variants'][name]={'forecast_errors_pixels':[float(np.linalg.norm(forecast[k]-reference[k])) for k in range(4)],
                  'forecast_before_observation':forecast.tolist(),'reference_xy':[reference[k].tolist() for k in range(4)],'outcomes':outcomes}
            trials.append(row);print('qualified rolling window',row['start_frame'],row['return_frame'],gap,flush=True)
    summary={}
    for stratum in ['all','early','middle','late']:
        subset=[r for r in trials if stratum=='all' or r['stratum']==stratum];summary[stratum]={}
        for name in VARIANTS:
            errors=[e for row in subset for e in row['variants'][name]['forecast_errors_pixels']]
            outcomes=[o for row in subset for o in row['variants'][name]['outcomes']]
            correct=sum(o['correct'] is True for o in outcomes);wrong=sum(o['correct'] is False for o in outcomes)
            summary[stratum][name]={'windows':len(subset),'cases':len(outcomes),'mean_forecast_pixels':float(np.mean(errors)) if errors else None,
               'median_forecast_pixels':float(np.median(errors)) if errors else None,'correct':correct,'wrong':wrong,'unknown':len(outcomes)-correct-wrong}
    assert all(sha(ROOT/p)==v for p,v in {**lock,**extra_lock}.items())
    write(REPORT/'rolling_results.json',{'attempted_windows':38,'qualified_windows':len(trials),'excluded_windows':excluded,
       'trials':trials,'summary':summary,'physical_ID_accuracy':None,'reference_is_local_FB_pseudo_tracklets':True,
       'global_seed_IDs_not_transferred_across_windows':True,'Gaussian_optimizer_steps':0,'all_SHA_unchanged':True})
    print('ROLLING COMPLETE',summary,flush=True)


if __name__=='__main__':main()
