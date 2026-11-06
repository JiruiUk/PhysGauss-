"""Task-local locks, immutable receipts and train RGB only."""
from pathlib import Path
from contextlib import contextmanager
import msvcrt
import numpy as np
from PIL import Image
from ..i10_tracking.io import ROOT, DATA, read, write, sha

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_pseudo_depth'
REPORT=BASE/'v1';RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_pseudo_depth/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_pseudo_depth/v1'

def stopped():
    return any(p.exists() for p in [ROOT/'stop.request',RUNS/'stop.request',RUNS.parent/'stop.request',RUNS.parent.parent/'stop.request'])

@contextmanager
def lock():
    assert not stopped();RUNS.mkdir(parents=True,exist_ok=True)
    with (RUNS/'audit.lock').open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):stream.write(b'0');stream.flush()
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)

def rgb(index):
    frame=read(DATA/'transforms_train.json')['frames'][index]
    # Same black-background compositing as the frozen PIDG train target. Alpha
    # never nominates an anchor, defines a class or filters an audit observation.
    a=np.asarray(Image.open(DATA/(frame['file_path']+'.png')).convert('RGBA').resize((400,225),Image.Resampling.BILINEAR))
    return np.uint8(a[...,:3].astype(np.float32)*a[...,3:4]/255)

def verify():
    for name,h in read(REPORT/'source_freeze.json')['files'].items():assert sha(ROOT/name)==h,name

def freeze():
    assert not (REPORT/'source_freeze.json').exists();p=read(BASE/'protocol.json')
    files={}
    receipts=['experiments/staged_reconstruction/RubberDuck_revocable_springs/v1/source_freeze.json',
              'experiments/staged_reconstruction/S1_pretrained_geometry/upstream_receipt.json']
    for name in receipts:
        value=read(ROOT/name)
        for path,h in value.get('files',value).items():
            assert sha(ROOT/path)==h,path;files[path]=h
        files[name]=sha(ROOT/name)
    paths=list(Path(__file__).parent.glob('*.py'))+list(BASE.glob('*.py'))
    paths += [BASE/'plan.md',BASE/'protocol.json',ROOT/'tests/test_i10_pseudo_depth.py',ROOT/p['tracks'],ROOT/p['visibility'],ROOT/p['trajectories']]
    paths += list((ROOT/'experiments/staged_reconstruction/RubberDuck_I10_tracking/dense/v1').glob('clip_*.npz'))
    paths += [ROOT/'variants/staged_reconstruction/pretrained_depth/model.py',ROOT/'variants/staged_reconstruction/pretrained_depth/core.py']
    for path in paths:files[path.relative_to(ROOT).as_posix()]=sha(path)
    assert files[p['parent']]==p['parent_sha256']
    write(REPORT/'source_freeze.json',dict(files=files,optimizer_updates=0,training_RGB_only=True));verify()
