"""Locked complete inference, or declared 4-view preflight recovery worker."""
import argparse
import os
import msvcrt
import numpy as np
import torch
from ..prepare_data import ROOT,read,write,sha
from ..synthetic_audit.input import load
from ..adaptive_static.geometry import patch_features
from ..pretrained_depth.model import configure
from ..rgb_prior_proposals.io import save_view,read_saved
from .prepare import BASE,REPORT,RUNS,PARENT,SOURCE,verify,stopped
from .compute import compute_view
from .inputs import load_parent,make_maps


def binding(case,i,train,row,sources):
    return {'case':case,'index':i,'sources':sources,'RGB_sha256':sha(train/row['image']),
        'camera_sha256':sha(train/'cameras.json'),'parent_D6_sha256':sha(PARENT/case/f'view_{i:03d}.npz'),
        'source_freeze_sha256':sha(REPORT/'source_freeze.json')}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--enable-plane-transport',action='store_true')
    p.add_argument('--preflight-directory',choices=['full','split']); p.add_argument('--first',type=int,default=0); p.add_argument('--last',type=int,default=96)
    args=p.parse_args()
    if not args.enable_plane_transport: raise SystemExit('Plane transport is opt-in; no inference files opened')
    assert not stopped(); verify()
    preflight=args.preflight_directory is not None
    if preflight:
        assert 0<=args.first<args.last<=4
        run=RUNS/'preflight'/args.preflight_directory
    else:
        assert args.first==0 and args.last==96 and not (REPORT/'inference.json').exists()
        assert read(REPORT/'preflight.json')['passed']; run=RUNS
    run.mkdir(parents=True,exist_ok=True)
    lock=(run/'inference.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    configure(); results={}
    try:
        protocol=read(BASE/'protocol.json')
        for case in protocol['synthetic_cases']:
            train=ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train'; values=load(train); meta=read(train/'cameras.json')
            source=read(SOURCE/f'{case}_inference.json'); near,far=source['camera_derived_depth_range']
            images=torch.tensor(values['rgb'],device='cuda').permute(0,3,1,2)
            features,texture=patch_features(images,5)
            views=torch.tensor(values['views'],dtype=torch.float32,device='cuda'); k=torch.tensor(values['K'],dtype=torch.float32,device='cuda')
            anchor_maps=make_maps(case,meta['height'],meta['width'])
            rows=[]
            for i in range(args.first,args.last):
                assert not stopped(),'Saved stop: preserve all view states and release task CUDA'
                row=meta['frames'][i]; path=run/case/f'view_{i:03d}.npz'
                bindings=binding(case,i,train,row,source['sources'][i])
                parent=load_parent(case,i); uv=parent['uv']; old=parent['updated']
                if path.exists(): receipt=read_saved(path,uv,old,bindings)
                else:
                    arrays,counts=compute_view(parent,anchor_maps,images,features,texture,views,k,i,source['sources'][i],near,far)
                    receipt=save_view(path,arrays,bindings,counts)
                rows.append(receipt['counts'])
                write(run/'state.json',{'status':'preflight' if preflight else 'inference','pid':os.getpid(),'case':case,'completed':i+1,'total':args.last})
                if preflight or i%12==0: print('PLANE_TRANSPORT',case,i+1,receipt['counts'],flush=True)
            results[case]={'views':len(rows),'totals':{key:sum(r[key] for r in rows) for key in rows[0]},'per_view':rows}
            del images,features,texture,views,k,anchor_maps; torch.cuda.empty_cache()
        verify()
        if not preflight:
            write(REPORT/'inference.json',{'GT_read':False,'Gaussian_training':False,'results':results,
                'source_freeze_sha256':sha(REPORT/'source_freeze.json'),'all_D6_candidates_preserved':True,
                'outputs':{p.relative_to(ROOT).as_posix():sha(p) for case in protocol['synthetic_cases'] for p in (run/case).glob('view_*') if p.is_file()}})
            write(run/'state.json',{'status':'inferred_pending_independent_evaluation'})
            print('PLANE_TRANSPORT_COMPLETE',{c:r['totals'] for c,r in results.items()},flush=True)
    except Exception as e:
        write(run/'state.json',{'status':'saved_stopped' if stopped() else 'failed','error':repr(e),'pid':os.getpid()}); raise
    finally:
        torch.cuda.empty_cache(); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()


if __name__=='__main__': main()
