"""Add immutable native-audit evidence; training decisions/losses are unchanged.

The original training sources are already frozen. This separately frozen I/O
adapter preserves each scheduled refresh model and all controller evidence for
independent native replay, without editing the frozen implementation.
"""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import torch
from variants.staged_reconstruction.i10_surface_graph import runtime as a
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,sha
from experiments.staged_reconstruction.RubberDuck_I10_surface_graph import launch

ORIGINAL_REFRESH=a.refresh
ORIGINAL_VERIFY=a.r.verify


def verify():
    ORIGINAL_VERIFY()
    for name,h in read(a.REPORT/'trace_freeze.json')['files'].items():assert sha(ROOT/name)==h,name


def refresh(renderer,scene,initial=False):
    assert not initial,'Formal trace uses the already frozen initial qualification'
    completed=a.STATE['calls'];directory=a.RUNS/'native_refresh'/a.ARM;directory.mkdir(parents=True,exist_ok=True)
    model=directory/f'model_{completed:04d}.pt';receipt=model.with_suffix('.json')
    assert not model.exists() and not receipt.exists()
    before=deepcopy(a.STATE)
    # This file is model-only audit evidence, explicitly not a resumable checkpoint.
    payload=dict(model=a.r.cpu_values(scene.state_dict()),completed_updates=completed,model_only_not_resume=True)
    tmp=model.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(model)
    summary=ORIGINAL_REFRESH(renderer,scene,False)
    a.r.atomic_json(receipt,dict(model_path=model.relative_to(ROOT).as_posix(),model_sha256=sha(model),
        controller_before=before,controller_after=deepcopy(a.STATE),summary=summary,
        source_sha256=sha(Path(__file__)),trace_freeze_sha256=sha(a.REPORT/'trace_freeze.json')))
    return summary


def worker(phase,arm='surface_graph',replicate=0):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=a.REPORT/f'{stamp}_{phase}_{arm}_{replicate}.log';path.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--enable-surface-graph','--phase',phase,'--arm',arm,'--replicate',str(replicate)]
    with path.open('x',encoding='utf-8') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,f'Preserve failed log {path}'
    return dict(log=path.relative_to(ROOT).as_posix(),sha256=sha(path))


if __name__=='__main__':
    a.refresh=refresh;a.r.verify=verify;launch.worker=worker;launch.main()
