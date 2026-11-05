"""Task-local artifacts, locks and frozen-input verification."""
from pathlib import Path
import hashlib
import json
import msvcrt
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_tracking'
REPORT=BASE/'v1'
RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_tracking/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_I10_tracking/v1'
DATA=ROOT/'data/derived/kinematic_resplit/training'
SOURCE=ROOT/'third_party/cotracker_research'
WEIGHTS=ROOT/'models/cotracker3/scaled_offline.pth'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stopped():
    return any(p.exists() for p in [ROOT/'stop.request',RUNS/'stop.request',RUNS.parent/'stop.request',RUNS.parent.parent/'stop.request'])
@contextmanager
def lock():
    assert not stopped(),'User stop request present'
    RUNS.mkdir(parents=True,exist_ok=True)
    with (RUNS/'pipeline.lock').open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):stream.write(b'0');stream.flush()
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
def verify():
    for name,h in read(REPORT/'source_freeze.json')['files'].items():assert sha(ROOT/name)==h,name
def freeze():
    assert not (REPORT/'source_freeze.json').exists(),'Preserve freeze'
    config=read(BASE/'protocol.json');meta=read(DATA/'transforms_train.json')
    assets=read(ROOT/'experiments/staged_reconstruction/learned_tracks/source_lock.json')
    assert assets['commit']==config['tracker_commit'] and sha(WEIGHTS)==config['tracker_sha256']
    files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'tests/test_i10_tracking.py',BASE/'protocol.json',BASE/'plan.md',DATA/'transforms_train.json',ROOT/config['parent'],WEIGHTS]
    files+=[DATA/(f['file_path']+'.png') for f in meta['frames']]
    files+=[SOURCE/name for name in assets['code_sha256']]
    files+=[ROOT/'output/short_clip_diagnostic/camera_math.py']
    inherited=read(ROOT/'experiments/staged_reconstruction/RubberDuck_I10_joint/v1/source_freeze.json')
    for name,h in inherited.get('files',inherited).items():
        assert sha(ROOT/name)==h,name
        files.append(ROOT/name)
    for folder in ['i10_joint','rubber_joint','union','joint_rgb','motion_integrity','state_recovery']:
        files+=list((ROOT/f'variants/staged_reconstruction/{folder}').glob('*.py'))
    # Parent construction/native camera imports, not color-based evidence paths.
    files+=[ROOT/f'variants/staged_reconstruction/{n}.py' for n in ['train_static','prepare_data','model','static_gaussian']]
    assert sha(ROOT/config['parent'])==config['parent_sha256']
    write(REPORT/'source_freeze.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))},
        new_evidence_no_appearance_mask_calls=True,optimizer_updates=0))
    verify()
