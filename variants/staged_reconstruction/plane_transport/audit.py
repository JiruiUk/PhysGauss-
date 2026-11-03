"""Independent process: all source/geometry/acceptance states and fixed RGB replay."""
from pathlib import Path
import numpy as np
import torch
from ..prepare_data import ROOT,read,write,sha
from ..synthetic_audit.input import load
from ..pretrained_depth.model import configure
from ..adaptive_static.geometry import patch_features
from .prepare import BASE,REPORT,RUNS,SOURCE,verify,stopped
from .inputs import load_parent,make_maps
from .infer import binding
from .core import query_hypotheses,incumbent_plane
from .compute import compute_view


def main():
    assert not stopped(); verify(); configure(); results={}
    receipt=read(REPORT/'inference.json')
    for name,h in receipt['outputs'].items(): assert sha(ROOT/name)==h,name
    for case in ['diffuse','glossy']:
        train=ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train'; meta=read(train/'cameras.json'); values=load(train)
        source=read(SOURCE/f'{case}_inference.json'); near,far=source['camera_derived_depth_range']
        views=torch.tensor(values['views'],dtype=torch.float32,device='cuda'); k=torch.tensor(values['K'],dtype=torch.float32,device='cuda')
        maps=make_maps(case,meta['height'],meta['width']); totals={key:0 for key in receipt['results'][case]['totals']}
        for i,row in enumerate(meta['frames']):
            assert not stopped(); parent=load_parent(case,i); uv,old=parent['uv'],parent['updated']
            path=RUNS/case/f'view_{i:03d}.npz'; saved_receipt=read(path.with_suffix('.json'))
            assert saved_receipt['output_sha256']==sha(path) and saved_receipt['bindings']==binding(case,i,train,row,source['sources'][i])
            with np.load(path) as a:
                assert np.array_equal(a['uv'],uv) and np.array_equal(a['original'],old)
                assert np.array_equal(a['parent_slope_admitted'],parent['slope_admitted'])
                baseline_valid=[]
                for start in range(0,len(old),256):
                    sl=slice(start,start+256); xy=torch.tensor(uv[sl],dtype=torch.float32,device='cuda'); z=torch.tensor(old[sl],device='cuda')
                    hyp=query_hypotheses(xy,z,maps,views,k,i,source['sources'][i],near,far,meta['height'],meta['width'])
                    for key,value in hyp.items(): assert np.array_equal(a[key][:,sl],value.cpu().numpy(),equal_nan=True),(case,i,key)
                    s,f,_,v=incumbent_plane(xy,z,torch.tensor(parent['coefficients'][sl,:2],dtype=torch.float32,device='cuda'),views[i],k,near,far)
                    assert np.array_equal(a['incumbent_slopes'][sl],s.cpu().numpy()) and np.array_equal(a['incumbent_fallback'][sl],f.cpu().numpy())
                    baseline_valid.append(v[0].cpu().numpy())
                which=a['candidate_fit'].argmin(0); best=a['candidate_fit'].min(0); rows=np.arange(len(old))
                eligible=a['candidate_valid'].any(0)&np.concatenate(baseline_valid)&np.isfinite(best)
                assert np.array_equal(a['eligible'],eligible)
                expected_source=np.where(eligible,np.asarray(source['sources'][i][:4])[which],-1)
                assert np.array_equal(a['selected_source'],expected_source)
                assert np.array_equal(a['selected_hypothesis'],np.where(eligible,which,-1))
                proposed=a['raw_candidate_depth'][which,rows]
                assert np.array_equal(a['proposed'],np.where(eligible,proposed,np.nan),equal_nan=True)
                assert np.array_equal(a['fit_new'],best,equal_nan=True)
                accepted=eligible&np.isfinite(proposed)&(proposed>0)&(proposed!=old)
                accepted&=np.isfinite(a['fit_old'])&np.isfinite(best)&(best<=a['fit_old']+1e-6)
                ho,hn=a['held_old'],a['held_new']
                accepted&=np.isfinite(ho).all(0)&np.isfinite(hn).all(0)&(hn<=ho+1e-6).all(0)&(hn<ho-1e-6).any(0)
                assert np.array_equal(accepted,a['accepted']) and np.array_equal(a['updated'],np.where(accepted,proposed,old))
                assert np.isfinite(a['updated']).all() and (a['updated']>0).all()
                counts={'candidates':len(old),'source_queries_admitted':int(a['source_query_valid'].sum()),
                    'valid_source_hypotheses':int(a['candidate_valid'].sum()),'eligible':int(eligible.sum()),
                    'eligible_without_target_slope':int((eligible&~parent['slope_admitted']).sum()),
                    'accepted':int(accepted.sum()),'accepted_without_target_slope':int((accepted&~parent['slope_admitted']).sum())}
                assert counts==saved_receipt['counts']
                for key,value in counts.items(): totals[key]+=value
        assert totals==receipt['results'][case]['totals']
        images=torch.tensor(values['rgb'],device='cuda').permute(0,3,1,2); features,texture=patch_features(images,5)
        for i in [0,24,72]:
            assert not stopped()
            arrays,_=compute_view(load_parent(case,i),maps,images,features,texture,views,k,i,source['sources'][i],near,far)
            with np.load(RUNS/case/f'view_{i:03d}.npz') as a:
                assert set(arrays)==set(a.files)
                for key,value in arrays.items(): assert np.array_equal(value,a[key],equal_nan=True),(case,i,key)
        results[case]={'totals':totals,'all96_parent_source_plane_query_geometry_and_acceptance_replay_exact':True,
            'full_raw_RGB_fixed_view_replay_exact':[0,24,72]}
        del images,features,texture,views,k,maps; torch.cuda.empty_cache()
    verify(); write(REPORT/'input_audit.json',{'passed':True,'GT_read':False,'results':results,
        'inference_sha256':sha(REPORT/'inference.json'),'source_sha256':sha(Path(__file__)),
        'source_original_intercepts_and_admission_unchanged':True,'all_candidate_membership_and_unaccepted_depths_exact':True})
    print('PLANE_TRANSPORT_INPUT_SOURCE_AND_RGB_AUDIT_PASSED',flush=True)


if __name__=='__main__': main()
