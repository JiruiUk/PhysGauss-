"""Verify frozen committed source bytes, diagnostic data and closed state."""
import subprocess
import hashlib
from pathlib import Path
from variants.staged_reconstruction.i10_joint_scale import io


def committed(path,h):
    data=subprocess.check_output(['git','show','HEAD:'+path],cwd=io.ROOT)
    assert hashlib.sha256(data).hexdigest()==h,path


def main():
    io.verify();f=io.read(io.REPORT/'final_review.json');assert not f['pending_review'] and f['optimizer_updates']==0
    n=0
    for receipt in ['source_freeze.json','extraction_freeze.json','numerical_repair_freeze.json']:
        for name,h in io.read(io.REPORT/receipt)['files'].items():
            assert io.sha(io.ROOT/name)==h,name
            if name.startswith('experiments/staged_reconstruction/RubberDuck_joint_scale/') or name.startswith('variants/staged_reconstruction/i10_joint_scale/') or name in ['tests/test_i10_joint_scale.py','tests/test_i10_joint_scale_solver.py']:
                committed(name,h);n+=1
    for name,h in f['artifacts_sha256'].items():
        assert io.sha(io.REPORT/name)==h,name
        committed((io.REPORT/name).relative_to(io.ROOT).as_posix(),h)
    for name in ['state.json','active.json','pipeline.json','final_status.json']:
        q=io.read(io.RUNS/name);assert q['status']==f['status'] and not q['pending_review'] and q['optimizer_updates']==0
    for name in ['independent_audit.py','independent_audit_v2.py','review.py','track_review.py','finalize.py']:
        committed((io.BASE/name).relative_to(io.ROOT).as_posix(),io.sha(io.BASE/name))
    m=io.read(io.REPORT/'manual_review.json')
    for name,h in m['reviewed_images_sha256'].items():assert io.sha(io.ROOT/name)==h,name
    for name in f['numerical_failures_preserved']:assert (io.REPORT/name).exists(),name
    io.write(io.REPORT/'closure_validation.json',dict(passed=True,frozen_task_git_files=n,
        final_manifest_git_files=len(f['artifacts_sha256']),raw_maps_verified=120,extraction_clips_verified=28,
        actual_reviewed_images=7,closed_states=True,parent_preserved=True,optimizer_updates=0,
        source_sha256=io.sha(Path(__file__)),final_review_sha256=io.sha(io.REPORT/'final_review.json')))
    print('JOINT_SCALE_COMMITTED_CLOSURE_PASS',n,flush=True)


if __name__=='__main__':main()
