"""Explicit opt-in; freeze and commit before extracting new RGB hypotheses."""
import argparse
from pathlib import Path
import numpy as np
from variants.staged_reconstruction.i10_joint_scale import io,core
from variants.staged_reconstruction.i10_joint_scale.audit import calculate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-joint-scale-audit',action='store_true')
    parser.add_argument('stage',choices=['freeze','extract','measure','replay']);args=parser.parse_args()
    if not args.enable_joint_scale_audit:raise SystemExit('Joint scale audit default off')
    with io.lock():
        if args.stage=='freeze':io.freeze();return
        io.verify();p=io.read(io.BASE/'protocol.json');tp=io.read(io.ROOT/p['tracking_protocol'])
        old=io.read(io.ROOT/p['old_protocol']);asset=np.load(io.ROOT/old['trajectories'])
        if args.stage in ['extract','replay']:
            images=[io.rgb(i) for i in range(120)]
            starts=p['starts'] if args.stage=='extract' else p['replay_clips']
            for start in starts:
                assert not io.stopped()
                file=io.RUNS/'clips'/f'{start:03d}.json';path=file.with_suffix('.npz')
                if args.stage=='extract' and file.exists():raise RuntimeError('Existing clip requires explicit interrupted-run review')
                arrays,result=core.extract(images[start:start+12],asset['views'][start:start+12],asset['k'],start,p,tp)
                if args.stage=='extract':
                    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays);io.write(file,result)
                    io.write(io.RUNS/'state.json',dict(status='extracting',last_start=start,optimizer_updates=0))
                    print('JOINT_SCALE_TRACKS',start,result['seeds'],result['qualified'],flush=True)
                else:
                    with np.load(path) as f:
                        for name,v in arrays.items():assert np.array_equal(v,f[name],equal_nan=True),(start,name)
                    assert result==io.read(file),start
            if args.stage=='extract':io.write(io.RUNS/'state.json',dict(status='extraction_complete',optimizer_updates=0))
        if args.stage in ['measure','replay']:
            result=calculate()
            if args.stage=='measure':
                assert not (io.REPORT/'audit.json').exists();io.write(io.REPORT/'audit.json',result)
                io.write(io.RUNS/'state.json',dict(status='completed_audit_pending_review',pending_review=True,optimizer_updates=0))
                print('JOINT_SCALE_RESULT',result['candidate']['summary'],flush=True)
            else:
                assert result==io.read(io.REPORT/'audit.json')
                io.write(io.REPORT/'independent_replay.json',dict(passed=True,clips=p['replay_clips'],
                    all120_decisions_exact=True,source_sha256=io.sha(Path(__file__)),optimizer_updates=0))
                print('JOINT_SCALE_EXACT_REPLAY_PASS',flush=True)
        io.verify()


if __name__=='__main__':main()
