"""One opt-in saved RGB depth refinement; no GT import or Gaussian optimization."""
import argparse
import msvcrt
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from ..prepare_data import ROOT,read,write,sha
from ..synthetic_audit.input import load
from ..adaptive_static.prepare import checked_maps
from ..adaptive_static.geometry import patch_features
from ..nuisance_static.prepare import verify as verify_parent
from .core import refine

REPORT = ROOT/'experiments/staged_reconstruction/S1_depth_evidence/refinement_v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_depth_refinement/v1'
SOURCE = ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2'
CACHE = ROOT/'runs/pidg_staged_reconstruction/S1_clean_visibility/synthetic_v2'


def verify():
    verify_parent()
    for p,h in read(REPORT/'source_freeze.json').items(): assert sha(ROOT/p) == h,p


def stopped():
    return (RUNS/'stop.request').exists() or (ROOT/'runs/pidg_staged_reconstruction/S1_nuisance/v1/stop.request').exists()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-depth-refinement',action='store_true'); args = parser.parse_args()
    if not args.enable_depth_refinement: raise SystemExit('Depth refinement is opt-in')
    assert not stopped(), 'Saved user stop; never auto-resume'
    verify(); assert read(REPORT/'tests.json')['passed']
    RUNS.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); cv2.setNumThreads(2)
    lock = (RUNS/'inference.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        protocol = read(SOURCE/'protocol.json'); results = {}
        for case in ['diffuse','glossy']:
            directory = RUNS/case; directory.mkdir(exist_ok=True)
            values = load(ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train')
            receipt = read(SOURCE/f'{case}_inference.json'); maps = []
            for i in range(96):
                with np.load(CACHE/case/f'raw_{i:03d}.npz') as data: maps.append(data['depth'].copy())
            checked,_,records = checked_maps(maps,receipt['sources'],values['views'],values['K'],
                np.rint(values['times']*200).astype(int).tolist(),protocol)
            assert records == receipt['per_view']
            image = torch.from_numpy(values['rgb']).permute(0,3,1,2).cuda()
            features,texture = patch_features(image,5); del image
            views = torch.tensor(values['views'],dtype=torch.float32,device='cuda'); k = torch.tensor(values['K'],dtype=torch.float32,device='cuda')
            near,far = receipt['camera_derived_depth_range']; counts = []
            for i,depth in enumerate(checked):
                assert not stopped(), 'Saved stop; release GPU and preserve per-view files'
                path = directory/f'view_{i:03d}.npz'
                if path.exists():
                    with np.load(path) as saved:
                        assert np.array_equal(saved['original'],depth)
                        counts.append(int(saved['updated'].sum()))
                    continue
                yy,xx = np.nonzero(depth > 0); uv = np.c_[xx,yy]; output = depth.copy(); updated = np.zeros_like(depth,dtype=bool)
                for start in range(0,len(uv),256):
                    xy = uv[start:start+256]; z = torch.tensor(depth[xy[:,1],xy[:,0]],device='cuda')
                    coord = torch.tensor(xy,dtype=torch.float32,device='cuda')
                    disabled,mask = refine(coord,z,features,texture,views,k,i,receipt['sources'][i],near,far,enabled=False)
                    assert disabled is z and not mask.any()
                    result,accepted = refine(coord,z,features,texture,views,k,i,receipt['sources'][i],near,far,enabled=True)
                    assert result.isfinite().all() and (result > 0).all()
                    output[xy[:,1],xy[:,0]] = result.cpu().numpy(); updated[xy[:,1],xy[:,0]] = accepted.cpu().numpy()
                assert np.array_equal(output > 0,depth > 0)
                assert np.array_equal(output[~updated],depth[~updated])
                np.savez_compressed(path,original=depth,refined=output,updated=updated)
                counts.append(int(updated.sum())); write(RUNS/'state.json',{'status':'inference','pid':os.getpid(),'case':case,'completed':i+1,'total':96})
                if i%12 == 0: print('DEPTH_REFINE',case,i+1,'updated',sum(counts),flush=True)
            results[case] = {'views':96,'accepted_updates':sum(counts),'original_candidates':int(sum((d > 0).sum() for d in checked)),
                'per_view_updates':counts,'membership_unchanged':True,'flags_off_exact':True}
            del features,texture; torch.cuda.empty_cache()
        verify(); write(REPORT/'inference.json',{'input_GT_read':False,'Gaussian_training':False,'results':results,
            'outputs':{p.relative_to(ROOT).as_posix():sha(p) for p in RUNS.rglob('view_*.npz')},'source_manifest_sha256':sha(REPORT/'source_freeze.json')})
        write(RUNS/'state.json',{'status':'inferred_pending_independent_evaluation'})
        print('DEPTH_REFINE_COMPLETED',results,flush=True)
    except Exception as error:
        write(RUNS/'state.json',{'status':'saved_stopped' if stopped() else 'failed','error':str(error),'pid':os.getpid()}); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()


if __name__ == '__main__': main()
