"""Independent-process membership, slope, acceptance and fixed raw-RGB replay."""
from pathlib import Path
import numpy as np
import torch
from ..prepare_data import ROOT,read,write,sha
from ..synthetic_audit.input import load
from ..pretrained_depth.model import configure
from ..adaptive_static.geometry import patch_features
from .prepare import BASE,REPORT,RUNS,D1,SOURCE,verify,stopped
from .infer import binding
from .core import fit_slopes
from .compute import compute_view


def main():
    assert not stopped(); verify(); configure(); results={}
    inference=read(REPORT/'inference.json'); protocol=read(BASE/'protocol.json')
    assert not inference['GT_read'] and not inference['Gaussian_training']
    for name,h in inference['outputs'].items(): assert sha(ROOT/name)==h,name
    for case in ['diffuse','glossy']:
        train=ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train'
        meta=read(train/'cameras.json'); source=read(SOURCE/f'{case}_inference.json')
        counts={k:0 for k in inference['results'][case]['totals']}; replayed=[]
        for i,row in enumerate(meta['frames']):
            assert not stopped()
            path=RUNS/case/f'view_{i:03d}.npz'; receipt=read(path.with_suffix('.json'))
            assert sha(path)==receipt['output_sha256']
            assert receipt['bindings']==binding(case,i,train,row,source['sources'][i])
            with np.load(D1/case/f'view_{i:03d}.npz') as a: depth=a['refined'].copy()
            yy,xx=np.nonzero(depth>0); uv=np.c_[xx,yy]; old=depth[yy,xx]
            expected=fit_slopes(depth,uv,protocol)
            with np.load(path) as a:
                assert np.array_equal(a['uv'],uv) and np.array_equal(a['original'],old)
                for key,value in expected.items(): assert np.array_equal(a[key],value,equal_nan=True),(case,i,key)
                new,proposed=a['updated'],a['proposed']; fitold,fitnew=a['fit_old'],a['fit_new']
                heldold,heldnew=a['held_old'],a['held_new']
                accepted=a['eligible']&np.isfinite(proposed)&(proposed>0)&(proposed!=old)
                accepted&=np.isfinite(fitold)&np.isfinite(fitnew)&(fitnew<=fitold+1e-6)
                accepted&=np.isfinite(heldold).all(0)&np.isfinite(heldnew).all(0)
                accepted&=(heldnew<=heldold+1e-6).all(0)&(heldnew<heldold-1e-6).any(0)
                assert np.array_equal(accepted,a['accepted'])
                assert np.array_equal(new,np.where(accepted,proposed,old))
                assert np.isfinite(new).all() and (new>0).all()
                actual={'candidates':len(old),'slope_admitted':int(a['slope_admitted'].sum()),
                    'eligible':int(a['eligible'].sum()),'finite_proposals':int(np.isfinite(proposed).sum()),'accepted':int(accepted.sum())}
                assert actual==receipt['counts']
            for key,value in actual.items(): counts[key]+=value
        assert counts==inference['results'][case]['totals']
        values=load(train); images=torch.tensor(values['rgb'],device='cuda').permute(0,3,1,2)
        features,texture=patch_features(images,5)
        views=torch.tensor(values['views'],dtype=torch.float32,device='cuda'); k=torch.tensor(values['K'],dtype=torch.float32,device='cuda')
        for i in [0,24,72]:
            assert not stopped()
            with np.load(D1/case/f'view_{i:03d}.npz') as a: depth=a['refined'].copy()
            arrays,actual=compute_view(depth,images,features,texture,views,k,i,source['sources'][i],*source['camera_derived_depth_range'],protocol)
            with np.load(RUNS/case/f'view_{i:03d}.npz') as saved:
                assert set(saved.files)==set(arrays)
                for key,value in arrays.items(): assert np.array_equal(saved[key],value,equal_nan=True),(case,i,key)
            replayed.append(i)
        results[case]={'all96_membership_input_slope_and_acceptance_replay':True,'totals':counts,'raw_RGB_fixed_full_view_replay_exact':replayed}
        del images,features,texture,views,k; torch.cuda.empty_cache()
    verify()
    write(REPORT/'input_audit.json',{'passed':True,'GT_read':False,'results':results,
        'inference_sha256':sha(REPORT/'inference.json'),'source_sha256':sha(Path(__file__)),
        'unaccepted_depths_and_all_candidate_membership_exact':True,'raw_RGB_full_fixed_views_all_fields_exact':True})
    print('RGB_PLANE_WARP_INPUT_AND_NATIVE_COST_AUDIT_PASSED',flush=True)


if __name__=='__main__': main()
