"""Own lock, stop requests and byte-frozen read-only inputs."""
from contextlib import contextmanager
from pathlib import Path
import msvcrt
import os
import traceback
from ..i10_pseudo_depth import io as old
from ..i10_tracking.io import ROOT, DATA, read, write, sha

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_joint_scale'
REPORT=BASE/'v1'
RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_joint_scale/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_joint_scale/v1'
rgb=old.rgb


def stopped():
    return old.stopped() or any(q.exists() for q in [RUNS/'stop.request',RUNS.parent/'stop.request'])


@contextmanager
def lock():
    assert not stopped(),'Stop request present; do not resume'
    RUNS.mkdir(parents=True,exist_ok=True)
    with (RUNS/'audit.lock').open('a+b') as f:
        f.seek(0)
        if not f.read(1):f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        write(RUNS/'active.json',dict(status='running',pid=os.getpid(),optimizer_updates=0))
        try:
            yield
            state=read(RUNS/'state.json') if (RUNS/'state.json').exists() else dict(status='frozen',optimizer_updates=0)
            write(RUNS/'active.json',dict(**state,pid=None))
        except BaseException:
            attempts=list(REPORT.glob('failure_attempt*.json'))
            write(REPORT/f'failure_attempt{len(attempts)+1}.json',dict(traceback=traceback.format_exc(),optimizer_updates=0))
            state=dict(status='paused' if stopped() else 'failed',optimizer_updates=0,pid=None)
            write(RUNS/'active.json',state);write(RUNS/'state.json',state)
            raise
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def verify():
    for name,h in read(REPORT/'source_freeze.json')['files'].items():assert sha(ROOT/name)==h,name


def freeze():
    assert not (REPORT/'source_freeze.json').exists()
    old.verify();files=read(old.REPORT/'source_freeze.json')['files'].copy()
    paths=list(Path(__file__).parent.glob('*.py'))+list(BASE.glob('*.py'))
    paths += [BASE/'protocol.json',BASE/'plan.md',ROOT/'tests/test_i10_joint_scale.py',
              old.REPORT/'audit.json',old.REPORT/'source_freeze.json',
              ROOT/'experiments/staged_reconstruction/RubberDuck_I10_tracking/dense/protocol.json']
    paths += list((old.RUNS/'raw').glob('*.npz'))
    assert len(list((old.RUNS/'raw').glob('*.npz')))==120
    for q in paths:files[q.relative_to(ROOT).as_posix()]=sha(q)
    write(REPORT/'source_freeze.json',dict(files=files,optimizer_updates=0,created_before_outputs=True))
    verify()
