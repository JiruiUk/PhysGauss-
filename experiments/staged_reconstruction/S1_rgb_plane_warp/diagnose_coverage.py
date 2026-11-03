"""Post-hoc evaluator-only localization of D6's limited correction coverage."""
from collections import Counter
from pathlib import Path
import numpy as np
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha
from variants.staged_reconstruction.pretrained_depth.evaluate import describe
from variants.staged_reconstruction.rgb_plane_warp.prepare import REPORT,RUNS


def main():
    assert read(REPORT/'input_audit.json')['passed']
    assert read(REPORT/'exports.json')['all_original_dataset_hashes_verified']
    results={}
    for case in ['diffuse','glossy']:
        data=ROOT/f'data/derived/clean_static_synthetic/v2/{case}'; meta=read(data/'train/cameras.json')
        counts=Counter(); angles={key:[] for key in ['admitted_visible','not_admitted_but_fitted_visible','admitted_visible_old_gross']}
        for i,row in enumerate(meta['frames']):
            with np.load(RUNS/case/f'view_{i:03d}.npz') as a:
                uv,old,admitted,accepted,coef=[a[key] for key in ['uv','original','slope_admitted','accepted','coefficients']]
            x,y=uv.T
            with np.load(data/'evaluation_only/gt'/f'{Path(row["image"]).stem}.npz') as a:
                truth=a['static_depth'][y,x]; visible=a['observed_static_visible'][y,x]&(truth>0); normals=a['static_normal'][y,x]
            gross=(truth<=0)|(np.abs(old-truth)>.05)
            counts.update({'candidates':len(old),'old_gross_or_no_static':int(gross.sum()),
                'old_gross_or_no_static_slope_admitted':int((gross&admitted).sum()),
                'old_gross_or_no_static_slope_rejected':int((gross&~admitted).sum()),
                'old_gross_or_no_static_accepted':int((gross&accepted).sum()),
                'visible_old_gross':int((gross&visible).sum()),
                'visible_old_gross_slope_admitted':int((gross&visible&admitted).sum()),
                'visible_old_gross_accepted':int((gross&visible&accepted).sum())})
            # Plane's inverse-depth covector -> camera normal -> world normal.
            covector=np.c_[coef[:,:2],1/old-coef[:,0]*x-coef[:,1]*y]
            ncam=covector@np.asarray(meta['K']); rotation=np.asarray(row['world_to_camera'])[:3,:3]
            nworld=ncam@rotation; norm=np.linalg.norm(nworld,axis=1)
            valid=visible&np.isfinite(nworld).all(1)&(norm>1e-12)&(np.linalg.norm(normals,axis=1)>0)
            nworld=nworld/np.maximum(norm[:,None],1e-12)
            theta=np.degrees(np.arccos(np.clip(np.abs((nworld*normals).sum(1)),0,1)))
            for key,mask in [('admitted_visible',admitted),('not_admitted_but_fitted_visible',~admitted),('admitted_visible_old_gross',admitted&gross)]:
                angles[key].append(theta[valid&mask])
        results[case]={'counts':dict(counts),'unoriented_normal_angle_degrees':{key:describe(np.concatenate(values)) for key,values in angles.items()}}
    write(REPORT/'coverage_diagnostic.json',{'post_hoc_evaluator_only':True,'used_to_change_D6_candidate_or_gate':False,
        'GT_pixel_arrays_or_training_normal_assets_exported':False,'results':results,'source_sha256':sha(Path(__file__)),
        'evaluation_sha256':sha(REPORT/'evaluation.json'),'inference_sha256':sha(REPORT/'inference.json'),
        'note':'Normals only where the least-squares coefficients exist; non-admitted but fitted excludes unfitted/unknown. Unoriented angles cannot establish physical correspondence.'})
    print('D6_POSTHOC_COVERAGE_DIAGNOSTIC',results,flush=True)


if __name__=='__main__': main()
