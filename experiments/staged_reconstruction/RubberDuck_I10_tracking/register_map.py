"""Append task-only cached map entries; preserve unrelated working edits."""
import difflib
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
SECTION='''## Mask-free I10 descendant persistent 3D tracking (T0-T2)

`variants/staged_reconstruction/i10_tracking/core.py` provides full-image
channel-symmetric texture proposals, bidirectional image gates, camera-motion
compensated stationary compatibility, persistent Gaussian-ID fitting/held
validation and separated-3D-alternative ambiguity. `extract.py` uses the pinned
local CoTracker3; no appearance or alpha foreground/static masks. `bind.py`
loads the immutable J6 I10-descendant state and checks native camera parity and
all 120 model-conditioned 3D trajectories. These are not measured 3D GT.
`i10_tracking_dense/run.py` isolates one denser proposal budget with unchanged
reliability/admission gates. T0 failed coverage; T1 passed with 123 tracks / 116
IDs. No physical object number or identity is inferred.

`i10_track_train/core.py` deduplicates persistent (frame,Gaussian ID) observations
and adds default-off robust 3D-to-image reprojection; `runtime.py` retains the
original UnionScene/deformation/native renderer and complete two Adam/RNG/
camera/physics state. Static background and physics are fully frozen; original
observed full RGB is retained. No old I10 target substitution, color-selected
static reference, foreground/union coverage term or new topology. Historical
color-conditioned parent lineage is disclosed. `RubberDuck_I10_tracking/
training/launch.py` tests native flags-off/gradient and independent exact-state/
fixed-native-gradient Adam recovery before matched 2000-step arms. Independent
native RGB/alpha/D, PLY and H264 evaluation is in task-local evaluate/audit files.
Tests: test_i10_tracking.py and test_i10_track_train.py. Original PIDG and all
frozen experiments remain untouched.'''

if __name__=='__main__':
    name='PIDG_REPO_MAP.md';path=ROOT/name
    head=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
    current=path.read_text(encoding='utf-8');assert SECTION.splitlines()[0] not in current
    path.write_text(current.rstrip()+'\n\n'+SECTION+'\n',encoding='utf-8',newline='\n')
    updated=head.rstrip()+'\n\n'+SECTION+'\n'
    patch=''.join(difflib.unified_diff(head.splitlines(True),updated.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    target=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_tracking/map_cached.patch';target.write_text(patch,encoding='utf-8',newline='\n')
    subprocess.run(['git','apply','--cached','--whitespace=nowarn',str(target)],cwd=ROOT,check=True)
