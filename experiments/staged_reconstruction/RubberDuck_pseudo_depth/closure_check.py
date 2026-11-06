"""Verify committed frozen bytes, all saved raw maps and closed audit states."""
from pathlib import Path
import hashlib
import subprocess
import numpy as np
from variants.staged_reconstruction.i10_pseudo_depth import io


def committed(path,h):
    raw=subprocess.check_output(['git','show','HEAD:'+path],cwd=io.ROOT)
    assert hashlib.sha256(raw).hexdigest()==h,path


def main():
    io.verify();final=io.read(io.REPORT/'final_review.json');assert not final['pending_review']
    assert final['optimizer_updates']==0 and not final['current_model_replaced']
    for name,h in final['artifacts_sha256'].items():
        assert io.sha(io.REPORT/name)==h,name
        committed((io.REPORT/name).relative_to(io.ROOT).as_posix(),h)
    count=0
    for path,h in io.read(io.REPORT/'source_freeze.json')['files'].items():
        if (path.startswith('variants/staged_reconstruction/i10_pseudo_depth/') or
            path.startswith('experiments/staged_reconstruction/RubberDuck_pseudo_depth/') or path=='tests/test_i10_pseudo_depth.py'):
            committed(path,h);count+=1
    a=io.read(io.REPORT/'audit.json')
    for i in range(120):
        path=io.RUNS/'raw'/f'{i:03d}.npz';assert io.sha(path)==a['raw_sha256'][str(i)]
        with np.load(path) as f:assert f['relative'].shape==(225,400) and np.isfinite(f['relative']).all()
    for path,h in io.read(io.REPORT/'manual_review.json')['reviewed_images_sha256'].items():assert io.sha(io.ROOT/path)==h,path
    for name in ['state.json','active.json','pipeline.json','final_status.json']:
        q=io.read(io.RUNS/name);assert q['status']==final['status'] and not q['pending_review'] and q['optimizer_updates']==0
    for filename,receipt in [('independent_audit.py','independent_evaluation_freeze.json'),
                             ('independent_audit_v2.py','independent_evaluation_freeze_v2.json'),
                             ('review_v2.py','figure_evaluation_freeze_v2.json')]:
        h=io.read(io.REPORT/receipt)['source_sha256'];assert io.sha(io.BASE/filename)==h
        committed((io.BASE/filename).relative_to(io.ROOT).as_posix(),h)
    io.write(io.REPORT/'closure_validation.json',dict(passed=True,frozen_task_Git_files=count,
        final_manifest_Git_files=len(final['artifacts_sha256']),raw_arrays_checked=120,actual_reviewed_images=6,
        closed_states=True,optimizer_updates=0,parent_preserved=True,
        source_sha256=io.sha(Path(__file__)),final_review_sha256=io.sha(io.REPORT/'final_review.json')))
    print('L7_COMMITTED_CLOSURE_PASS',count,120,6)


if __name__=='__main__':main()
