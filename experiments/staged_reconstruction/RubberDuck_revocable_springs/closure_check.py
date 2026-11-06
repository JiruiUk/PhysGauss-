"""Independent final committed-byte, checkpoint, media and closed-state check."""
from pathlib import Path
import hashlib
import subprocess
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

def main():
    b=Path(__file__).parent;v=b/'v1';r=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_revocable_springs/v1'
    final=read(v/'final_review.json');assert not final['pending_review']
    for name,h in final['artifacts_sha256'].items():
        assert sha(v/name)==h,name
        path=(v/name).relative_to(ROOT).as_posix()
        raw=subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==h,path
    manual=read(v/'manual_review.json');assert len(manual['reviewed_images_sha256'])==32
    for path,h in manual['reviewed_images_sha256'].items():assert sha(ROOT/path)==h,path
    for name in ['final_status.json','pipeline.json','active.json']:
        q=read(r/name);assert q['status']==final['status'] and not q['pending_review']
    for arm in ['temporal_control','revocable_spring']:
        q=read(r/'formal'/arm/'final_status.json');assert q['iteration']==2000 and not q['pending_review']
        assert q['checkpoint_sha256']==sha(r/'formal'/arm/'checkpoint.pt')
    count=0
    for receipt in ['source_freeze.json','evaluation_freeze.json']:
        values=read(v/receipt);values=values.get('files',values)
        for path,h in values.items():
            assert sha(ROOT/path)==h,path
            if (path.startswith('variants/staged_reconstruction/i10_springs/') or
                path.startswith('experiments/staged_reconstruction/RubberDuck_revocable_springs/') or path=='tests/test_i10_springs.py'):
                raw=subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)
                assert hashlib.sha256(raw).hexdigest()==h,path
                count+=1
    write(v/'closure_validation.json',dict(passed=True,actual_reviewed_images=32,
        full_checkpoints_checked=2,frozen_task_Git_checks=count,final_manifest_checked=len(final['artifacts_sha256']),
        final_manifest_Git_checked=len(final['artifacts_sha256']),final_review_sha256=sha(v/'final_review.json'),
        closed_states=True,no_training_modified=True,source_sha256=sha(Path(__file__))))
    print('L6_CLOSURE_PASS',count,'frozen Git checks,32 actual image checks,two complete checkpoints')

if __name__=='__main__':main()
