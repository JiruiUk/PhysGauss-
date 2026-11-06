"""Default-off serial audit, preserving raw maps and failed attempts."""
from pathlib import Path
import argparse
import numpy as np
from variants.staged_reconstruction.i10_pseudo_depth import io
from variants.staged_reconstruction.pretrained_depth.model import load_model,predict
from variants.staged_reconstruction.i10_pseudo_depth.audit import calculate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-pseudo-depth-audit',action='store_true')
    parser.add_argument('stage',choices=['freeze','infer','measure','replay']);args=parser.parse_args()
    if not args.enable_pseudo_depth_audit:raise SystemExit('Pseudo-depth audit default off')
    with io.lock():
        if args.stage=='freeze':io.freeze();return
        io.verify();p=io.read(io.BASE/'protocol.json')
        if args.stage in ['infer','replay']:
            model=load_model(io.ROOT,p,True)
            indices=range(120) if args.stage=='infer' else p['fixed_review_frames']
            values=[]
            for i in indices:
                assert not io.stopped()
                path=io.RUNS/'raw'/f'{i:03d}.npz'
                if args.stage=='infer' and path.exists():raise RuntimeError('Preserve saved raw output; use explicit interrupted resume review')
                raw=predict(None,io.rgb(i),model,p['official_input_size'],True)
                if args.stage=='infer':
                    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,relative=raw)
                    io.write(io.RUNS/'state.json',dict(status='inference',completed=i+1,total=120,optimizer_updates=0))
                    print('L7_PSEUDO_DEPTH',i+1,120,flush=True)
                else:
                    with np.load(path) as f:assert np.array_equal(f['relative'],raw),i
                    values.append(i)
            del model
            if args.stage=='replay':
                assert calculate()==io.read(io.REPORT/'audit.json')
                io.write(io.REPORT/'independent_replay.json',dict(passed=True,all120_CPU_decisions_exact=True,
                    independently_reloaded_raw_frames_exact=values,all120_raw_SHA_verified=True,
                    optimizer_updates=0,source_sha256=io.sha(Path(__file__))))
                print('L7_FULL_CPU_AND_FIXED_RAW_REPLAY_PASS',flush=True)
        if args.stage=='measure':
            assert not (io.REPORT/'audit.json').exists();result=calculate()
            io.write(io.REPORT/'audit.json',result)
            io.write(io.RUNS/'state.json',dict(status='audit_complete_pending_review',pending_review=True,**result['summary']))
            print('L7_AUDIT',result['summary'],flush=True)
        io.verify()


if __name__=='__main__':main()
