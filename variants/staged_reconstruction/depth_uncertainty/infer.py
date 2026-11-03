"""RGB-derived source-depth intervals; never opens evaluator GT."""
import argparse
import msvcrt
import os
import numpy as np
from ..prepare_data import ROOT,read,write,sha
from ..synthetic_audit.input import load
from .prepare import BASE,REPORT,RUNS,PARENT,SOURCE,verify,stopped
from .core import add_evidence


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-depth-uncertainty',action='store_true'); args = parser.parse_args()
    if not args.enable_depth_uncertainty: raise SystemExit('Depth interval evidence is opt-in')
    assert not stopped(), 'Saved user stop; never auto-resume'
    verify(); assert read(REPORT/'test_receipt.json')['passed']
    assert not (REPORT/'inference.json').exists(), 'Preserve completed candidate'
    RUNS.mkdir(parents=True,exist_ok=True)
    lock = (RUNS/'inference.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        protocol = read(BASE/'protocol.json'); results = {}
        for case in protocol['synthetic_cases']:
            directory = RUNS/case; directory.mkdir(exist_ok=True)
            values = load(ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train')
            metadata = read(SOURCE/f'{case}_inference.json'); depths = []
            for i in range(96):
                with np.load(PARENT/case/f'view_{i:03d}.npz') as data: depths.append(data['refined'].copy())
            frame_ids = np.rint(values['times']*200).astype(int)
            near,far = metadata['camera_derived_depth_range']; counts = []
            for i,depth in enumerate(depths):
                assert not stopped(), 'Saved stop; preserve per-view state'
                yy,xx = np.nonzero(depth > 0); uv = np.c_[xx,yy]; z = depth[yy,xx]
                path = directory/f'view_{i:03d}.npz'
                if path.exists():
                    with np.load(path) as saved:
                        assert np.array_equal(saved['uv'],uv) and np.array_equal(saved['depth'],z)
                        counts.append(int(saved['tight'].sum()))
                    continue
                unchanged,disabled = add_evidence(z)
                assert unchanged is z and disabled is None
                unchanged,evidence = add_evidence(z,uv,values['views'],values['K'],i,metadata['sources'][i],
                    depths,frame_ids,near,far,protocol,True)
                assert unchanged is z and np.array_equal(unchanged,depth[yy,xx])
                np.savez_compressed(path,uv=uv,depth=z,**evidence)
                counts.append(int(evidence['tight'].sum()))
                write(RUNS/'state.json',{'status':'inference','pid':os.getpid(),'case':case,'completed':i+1,'total':96})
                if i%12 == 0: print('DEPTH_UNCERTAINTY',case,i+1,'tight',sum(counts),flush=True)
            results[case] = {'views':96,'candidates':int(sum((d > 0).sum() for d in depths)),
                'tight_candidates':sum(counts),'per_view_tight':counts,'all_candidates_and_depth_unchanged':True}
        verify(); write(REPORT/'inference.json',{'GT_read':False,'Gaussian_training':False,'results':results,
            'outputs':{p.relative_to(ROOT).as_posix():sha(p) for p in RUNS.rglob('view_*.npz')},
            'empirical_envelope_not_posterior':True,'source_manifest_sha256':sha(REPORT/'source_freeze.json')})
        write(RUNS/'state.json',{'status':'inferred_pending_independent_evaluation'})
        print('DEPTH_UNCERTAINTY_COMPLETE',results,flush=True)
    except Exception as error:
        write(RUNS/'state.json',{'status':'saved_stopped' if stopped() else 'failed','error':str(error),'pid':os.getpid()}); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()


if __name__ == '__main__': main()
