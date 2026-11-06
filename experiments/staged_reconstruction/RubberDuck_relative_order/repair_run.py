"""Independent metadata-only refresh repair over immutable training sources."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from experiments.staged_reconstruction.RubberDuck_relative_order import launch as original
from variants.staged_reconstruction.i10_relative_order import runtime as a
from variants.staged_reconstruction.i10_ordinal_depth.core import transition
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,sha

VERIFY=a.r.verify


def clean_transition(previous,known,passed,initial=False):
    # Measurement is evidence metadata, not part of the hysteresis state.
    state=None if previous is None else {k:v for k,v in previous.items() if k!='measurement'}
    return transition(state,known,passed,initial)


def verify():
    VERIFY()
    for name,value in read(a.REPORT/'repair_freeze.json')['files'].items():assert sha(ROOT/name)==value,name


def worker(phase,arm='ordinal_depth',replicate=0):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=a.REPORT/f'{stamp}_repair_{phase}_{arm}_{replicate}.log'
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--enable-ordinal-depth','--phase',phase,'--arm',arm,'--replicate',str(replicate)]
    with path.open('x',encoding='utf-8') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,f'Preserve failed repair log: {path}'
    return dict(log=path.relative_to(ROOT).as_posix(),sha256=sha(path))


def install():
    a.q.transition=clean_transition;a.r.verify=verify;original.worker=worker


if __name__=='__main__':
    install();original.main()
