"""Separate read-only replay of exact metadata and fixed original RGB sources."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import numpy as np
import torch
from variants.staged_reconstruction.rubber_visibility import worker as w
from .qualification_repair import verify_supplement


def independent_statistics(weight, observed):
    a=weight.astype(np.float64);b=observed.astype(np.float64).mean(axis=2)
    total=float(np.sum(a))
    return {'visible_weight':total,
        'observed_mean_RGB':None if total==0 else float(np.sum(a*b)/total),
        'weight_on_observed_RGB_lt005':None if total==0 else float(np.sum(a[b<.05])/total)}


@torch.no_grad()
def review():
    verify_supplement();assert not w.stopped()
    assert not (w.REPORT/'independent_audit.json').exists()
    manifest=w.REPORT/'independent_source_freeze.json'
    assert not manifest.exists()
    w.old.atomic_json(manifest,{w.Path(__file__).relative_to(w.ROOT).as_posix():w.sha(w.Path(__file__))})
    data=w.read(w.REPORT/'audit.json');meta=w.read(w.DATA/'transforms_train.json')
    times=defaultdict(list);cameras=defaultdict(set)
    for frame in meta['frames']:
        key=hashlib.sha256(json.dumps(frame['transform_matrix'],separators=(',',':')).encode()).hexdigest()
        times[float(frame['time']).hex()].append(key);cameras[key].add(float(frame['time']).hex())
    census={'frames':len(meta['frames']),'unique_times':len(times),'unique_camera_transforms':len(cameras),
        'time_camera_counts':[{'time_hex':t,'images':len(v),'unique_camera_transforms':len(set(v))}
            for t,v in sorted(times.items(),key=lambda pair:float.fromhex(pair[0]))],
        'same_time_multicamera_times':sum(len(set(v))>1 for v in times.values()),
        'camera_time_counts':sorted(len(v) for v in cameras.values())}
    assert census==data['metadata']
    for item in data['fixed_time_coverage']:
        distances=[abs(float(f['time'])-(item['target_number']-1)/199) for f in meta['frames']]
        assert item['exact_train_images']==distances.count(0.)
        assert item['nearest_train_time_gap']==min(distances)
        assert item['nearest_training_files']==[f['file_path'] for f,d in zip(meta['frames'],distances) if d==min(distances)]
    byid={int(w.Path(f['file_path']).stem):f for f in meta['frames']};results={}
    torch.set_num_threads(4)
    for name in data['models']:
        assert not w.stopped();renderer,scene,opts,state,step,trace,before=w.load(name)
        population=len(scene.pools[0].get_xyz);static_count=len(scene.static.get_xyz)
        groups=[];target_checks=[]
        for item in data['models'][name]['targets']:
            cam=w.camera(byid[item['camera']],meta,1600);t=(item['number']-1)/199
            package=w.native(renderer,scene,cam,t,dynamic_only=True)
            rgb=package['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
            alpha=package['alpha'].squeeze().cpu().numpy();mask=(alpha>.5)&(rgb.mean(axis=2)<.05)
            values=package['gs_per_pixel'].cpu().numpy()[:,mask].ravel()
            selected=np.unique(values[(values>=0)&(values<population)]).astype(np.int64)
            colors=torch.zeros((population,3),device='cuda');colors[selected,0]=1
            weights=w.native(renderer,scene,cam,t,colors,dynamic_only=True)['render'][0].clamp_min(0).cpu().numpy()
            denominator=float(alpha[mask].astype(np.float64).sum());mass=float(weights[mask].astype(np.float64).sum())
            computed={'pixels':mask.size,'dynamic_alpha_gt05_pixels':int((alpha>.5).sum()),
                'dark_high_alpha_pixels':int(mask.sum()),'selected_canonical_rows':len(selected),
                'target_dark_alpha_mass':denominator,'selected_native_visible_mass_on_dark_set':mass,
                'fraction_dark_alpha_explained_by_first20_ID_union':None if denominator==0 else mass/denominator}
            for key,value in computed.items():assert value==item[key],(name,key,value,item[key])
            groups.append(selected);target_checks.append({'camera':item['camera'],'number':item['number'],'exact':True})
        source_checks=[]
        for index in [0,24,72]:
            assert not w.stopped();frame=meta['frames'][index];cam=w.camera(frame,meta,1600)
            observed=w.old.frame_target(frame).permute(1,2,0).cpu().numpy();rows=[]
            for offset in [0,3]:
                colors=torch.zeros((static_count+population,3),device='cuda')
                for channel in range(3):colors[static_count+groups[offset+channel],channel]=1
                package=w.native(renderer,scene,cam,frame['time'],colors)
                for channel in range(3):
                    weight=package['render'][channel].clamp_min(0).cpu().numpy()
                    rows.append(independent_statistics(weight,observed))
            saved=data['models'][name]['sources'][index]
            assert saved['file_path']==frame['file_path'] and saved['time']==frame['time']
            w.old.assert_equivalent(rows,saved['groups'],0,0)
            source_checks.append({'index':index,'six_native_group_statistics_exact':True})
        # Recompute every aggregate from all saved source denominators, without model selection.
        sources=data['models'][name]['sources'];assert len(sources)==120
        for group,aggregate in enumerate(data['models'][name]['source_totals']):
            values=[row['groups'][group] for row in sources];total=sum(v['visible_weight'] for v in values)
            assert total==aggregate['total_visible_weight']
            assert sum(v['visible_weight']>0 for v in values)==aggregate['visible_nonzero_sources']
            for source_key,target_key in [('observed_mean_RGB','weighted_observed_mean_RGB'),
                    ('weight_on_observed_RGB_lt005','weight_on_observed_RGB_lt005')]:
                result=None if total==0 else sum(v['visible_weight']*(v[source_key] or 0) for v in values)/total
                assert result==aggregate[target_key]
        w.assert_no_update(scene,opts,state,step,trace,before)
        results[name]={'targets':target_checks,'sources':source_checks,'all120_aggregate_denominators_exact':True,
            'full_state_no_update_exact':True}
        print('J5_INDEPENDENT_PASS',name,flush=True)
        del scene,opts,before,renderer;gc.collect();torch.cuda.empty_cache()
    for path,h in data['images'].items():assert w.sha(w.ROOT/path)==h,path
    verify_supplement()
    for path,h in w.read(manifest).items():assert w.sha(w.ROOT/path)==h
    w.old.atomic_json(w.REPORT/'independent_audit.json',{'passed':True,'metadata_exact':True,
        'all_image_hashes_match':True,'models':results,'training_updates':0,
        'audit_sha256':w.sha(w.REPORT/'audit.json'),'source_sha256':w.sha(w.Path(__file__))})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-rubber-visibility-audit',action='store_true');args=parser.parse_args()
    if not args.enable_rubber_visibility_audit:raise SystemExit('J5 independent review default off; no input opened')
    w.old.RUNS=w.RUNS
    with w.old.lock('worker.lock'):review()


if __name__=='__main__':main()
