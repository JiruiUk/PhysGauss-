"""Add this branch to the committed map without staging parallel map edits."""
import difflib
import subprocess
from variants.staged_reconstruction.rubber_joint import runtime as r
from variants.staged_reconstruction.prepare_data import ROOT

SECTION='''## Default-off Rubber Duck paired RGB joint pilot (J4)

`variants/staged_reconstruction/rubber_joint/core.py` opens only static opacity/SH
when explicitly enabled; the original `UnionScene.render` and frozen PIDG
renderer remain untouched. It reuses the frozen J3 opacity-only parent-depth
self-reference. `runtime.py` loads the pinned complete S3-U parent plus the
exactly matching S1 static Adam, retains complete dynamic Adam/RNG/sampler and
physics graphs, freezes velocity/stress, and saves/restores both optimizers and
bindings. This is RGB reconstruction, not original physics continuation; inherited
color-mask training lineage is disclosed. `RubberDuck_joint_pilot/launch.py`
freezes train inputs, qualifies original native output/gradients and independent
exact full-state/fixed-gradient Adam recovery before serial matched2000 updates.
`evaluate.py` independently audits all initial/final states and30 validation
images, native fixed86/188 early/mid/late RGB/alpha/D, PLY and fully decoded
H264. Fixed-view cross-time has no independent GT; teacher is a self-reference.
Tests: `tests/test_rubber_joint.py`; default off has original object identity.
No topology/identity/semantic masks/GT or physics modules are added.'''

def main():
    name='PIDG_REPO_MAP.md';head=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
    path=ROOT/name;current=path.read_text(encoding='utf-8');assert SECTION.splitlines()[0] not in current
    path.write_text(current.rstrip()+'\n\n'+SECTION+'\n',encoding='utf-8',newline='\n')
    updated=head.rstrip()+'\n\n'+SECTION+'\n'
    patch=''.join(difflib.unified_diff(head.splitlines(True),updated.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    dest=r.RUNS/'map_cached.patch';dest.write_text(patch,encoding='utf-8',newline='\n');print(dest)

if __name__=='__main__':main()
