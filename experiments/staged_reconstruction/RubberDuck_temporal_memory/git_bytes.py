"""Preserve frozen raw JSON bytes in Git; never modify the worktree inputs."""
import hashlib
import subprocess
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

def main():
    b=Path(__file__).parent;entries={}
    for name in ['source_freeze.json','evaluation_freeze.json']:
        values=read(b/'v1'/name);entries.update(values.get('files',values))
    checked=[];fixed=[]
    for path,h in entries.items():
        assert sha(ROOT/path)==h,path
        if not (path.startswith('experiments/staged_reconstruction/RubberDuck_temporal_memory/') or
                path.startswith('variants/staged_reconstruction/i10_temporal_memory/') or path=='tests/test_i10_temporal_memory.py'):
            continue
        raw=subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=h:
            blob=subprocess.check_output(['git','hash-object','-w','--stdin'],input=(ROOT/path).read_bytes(),cwd=ROOT).decode().strip()
            subprocess.run(['git','update-index','--cacheinfo','100644,'+blob+','+path],cwd=ROOT,check=True)
            fixed.append(path)
        checked.append(path)
    write(b/'v1/git_validation_attempt1.json',dict(passed=False,reason='Git autocrlf normalized frozen bank/health JSON bytes; worktree frozen hashes still exact',frozen_training_or_inputs_modified=False))
    write(b/'v1/git_storage_repair.json',dict(raw_bytes_staged=fixed,checked=checked,worktree_or_training_modified=False))
    print('Staged exact raw bytes only:',fixed)

if __name__=='__main__':main()
