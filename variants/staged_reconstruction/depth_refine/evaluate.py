"""Independent GT audit of fixed-candidate depth refinement, no training writes."""
from pathlib import Path
import sys
import numpy as np
from ..prepare_data import ROOT,read,write,sha
from .infer import REPORT,RUNS,verify


def main():
    verify(); assert not (REPORT/'evaluation.json').exists(), 'Preserve completed evaluation'
    receipt = read(REPORT/'inference.json')
    for p,h in receipt['outputs'].items(): assert sha(ROOT/p) == h,p
    sys.path.insert(0,str(REPORT.parent))
    from audit import normal_values,describe
    results = {}
    for case in ['diffuse','glossy']:
        private = ROOT/f'data/derived/clean_static_synthetic/v2/{case}'
        meta = read(private/'train/cameras.json'); k = np.asarray(meta['K']); buckets = {}
        for arm in ['original','refined']:
            buckets[arm] = {name:[] for name in ['static_error','visible_static_error','thin_error','normal_deg_fixed_original_pixels']}
            buckets[arm].update({'same_ray_005':0,'same_ray_002':0,'candidates':0,'normal_available':0})
        changes = []
        for i,row in enumerate(meta['frames']):
            with np.load(RUNS/case/f'view_{i:03d}.npz') as saved:
                original,refined,updated = [saved[n].copy() for n in ['original','refined','updated']]
            assert np.array_equal(original > 0,refined > 0)
            assert np.array_equal(original[~updated],refined[~updated])
            changes.append(np.abs(refined[updated]-original[updated]))
            view = np.asarray(row['world_to_camera'])
            _,eligible = normal_values(original,view,k,.025)
            with np.load(private/'evaluation_only/gt'/(Path(row['image']).stem+'.npz')) as gt:
                depth,normal = gt['static_depth'],gt['static_normal']; surface = depth > 0
                face = surface[1:-1,1:-1].copy()
                for ay,ax in [(slice(1,-1),slice(2,None)),(slice(1,-1),slice(None,-2)),
                        (slice(2,None),slice(1,-1)),(slice(None,-2),slice(1,-1))]:
                    face &= surface[ay,ax]&((normal[1:-1,1:-1]*normal[ay,ax]).sum(2) > .99)
                fixed = eligible&face
                for arm,values in [('original',original),('refined',refined)]:
                    bucket = buckets[arm]; selected = values > 0; on = selected&surface; error = np.abs(values-depth)
                    bucket['candidates'] += int(selected.sum()); bucket['same_ray_005'] += int((on&(error <= .05)).sum())
                    bucket['same_ray_002'] += int((on&(error <= .02)).sum())
                    bucket['static_error'].append(error[on]); bucket['visible_static_error'].append(error[selected&gt['observed_static_visible']&surface])
                    bucket['thin_error'].append(error[selected&gt['thin_static_visible']&surface])
                    predicted,good = normal_values(values,view,k,.025)
                    angle = np.degrees(np.arccos(np.clip(np.abs((predicted*normal[1:-1,1:-1]).sum(2)),0,1)))
                    bucket['normal_deg_fixed_original_pixels'].append(angle[fixed]); bucket['normal_available'] += int(good.sum())
        for arm,bucket in buckets.items():
            for key in ['static_error','visible_static_error','thin_error','normal_deg_fixed_original_pixels']:
                bucket[key] = describe(np.concatenate(bucket[key]))
            bucket['same_ray_005_fraction'] = bucket['same_ray_005']/bucket['candidates']
            bucket['same_ray_002_fraction'] = bucket['same_ray_002']/bucket['candidates']
        assert buckets['original']['candidates'] == buckets['refined']['candidates']
        results[case] = {'arms':buckets,'absolute_depth_update':describe(np.concatenate(changes)),
            'membership_and_unupdated_depth_exact':True,'normal_comparison_fixed_original_validity':True}
    for p,h in receipt['outputs'].items(): assert sha(ROOT/p) == h,p
    verify(); write(REPORT/'evaluation.json',{'results':results,'inference_GT_read':False,'independent_GT_evaluation':True,
        'complete_Gaussian_training':False,'real_continuation_authorized':False,'inference_receipt_sha256':sha(REPORT/'inference.json'),
        'caution':'Held RGB sources participate in acceptance, not independent evaluation; no promotion from mean alone.'})
    print('DEPTH_REFINEMENT_EVALUATED',results,flush=True)


if __name__ == '__main__': main()
