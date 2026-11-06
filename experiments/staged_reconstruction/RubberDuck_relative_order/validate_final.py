"""Independent closure/receipt checks, no rendering or model updates."""
import argparse
import hashlib
import msvcrt
from pathlib import Path
import subprocess
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

B=ROOT/'experiments/staged_reconstruction/RubberDuck_relative_order'
R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_relative_order/v1'


def run():
    frozen={}
    for receipt in ['preregistration.json','v1/source_freeze.json','v1/evaluation_freeze.json','v1/repair_freeze.json']:
        obj=read(B/receipt)
        for name,h in obj.get('files',obj).items():assert sha(ROOT/name)==h,name;frozen[name]=h
    committed=[]
    for name,h in frozen.items():
        if name.startswith(('variants/staged_reconstruction/i10_relative_order/',
                            'experiments/staged_reconstruction/RubberDuck_relative_order/',
                            'tests/test_i10_relative_order')):
            result=subprocess.run(['git','show','HEAD:'+name],cwd=ROOT,capture_output=True)
            assert result.returncode==0 and hashlib.sha256(result.stdout).hexdigest()==h,name
            committed.append(name)
    for name in ['pairs_audit','controller_audit','independent_audit','recovery','native_check']:
        assert read(B/f'v1/{name}.json')['passed'],name
    for doc in ['pairs_audit','controller_audit']:
        assert read(B/f'v1/{doc}.json')['source_sha256']==sha(B/'evidence_audit.py')
    assert read(B/'v1/independent_audit.json')['source_sha256']==sha(B/'audit.py')
    ev=read(B/'v1/evaluation.json');final=read(B/'v1/final_review.json');manual=read(B/'v1/manual_review.json')
    assert ev['source_sha256']==sha(B/'evaluate.py')
    assert final['source_sha256']==sha(B/'close.py') and final['manual_review_sha256']==sha(B/'v1/manual_review.json')
    assert not final['pending'] and not final['pending_review'] and not final['promoted'] and final['retained_model']=='J6'
    assert not manual['cleanup_pass'] and not manual['completeness_pass']
    for name,h in manual['viewed_images'].items():assert sha(ROOT/name)==h
    receipt=read(B/'v1/training_receipt.json');assert receipt['retained_updates']==1200
    for name,h in receipt['failure_preserved'].items():assert sha(ROOT/name)==h,name
    for q in receipt['training_logs']:assert sha(ROOT/q['log'])==q['sha256']
    for arm in ['rgb_control','ordinal_depth']:
        d=R/'formal'/arm;state=read(d/'final_status.json');saved=read(d/'checkpoint.json')
        assert state['iteration']==saved['iteration']==600 and not state['pending_review']
        assert state['checkpoint_sha256']==saved['sha256']==sha(d/'checkpoint.pt')
        assert receipt['arms'][arm]['final']==saved
        assert receipt['arms'][arm]['usage_sha256']==sha(d/'usage.json')
        assert state['review_sha256']==sha(B/'v1/final_review.json')
    for name in ['pipeline','active','final_status']:
        q=read(R/f'{name}.json');assert q['status']==final['status'] and not q['pending'] and not q['pending_review']
        assert not q['optimizer_running'] and q['review_sha256']==sha(B/'v1/final_review.json')
    order=ev['ordinal_diagnostic']['models']
    for split,n,agreement in [('fit',30,9),('held',32,19)]:
        for model in order:assert order[model][split]['denominator']==n and order[model][split]['order_correct']==agreement
        a={q['id']:q for q in order['rgb_control'][split]['rows']};b={q['id']:q for q in order['ordinal_depth'][split]['rows']}
        assert set(a)==set(b) and all(a[k]['valid']==b[k]['valid'] and a[k]['order_correct']==b[k]['order_correct'] for k in a)
    locks=[]
    for name in ['pipeline.lock','train.lock','qualification.lock','evaluation.lock']:
        with (R/name).open('r+b') as f:
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        locks.append(name)
    for name in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']:
        raw=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
        assert '### L8-R Rubber Duck局部相对顺序：匹配训练完成' in raw
    write(B/'v1/final_validation.json',dict(passed=True,frozen_files_verified=len(frozen),
        task_frozen_committed_raw_bytes_exact=committed,all_final_statuses_closed=True,all_task_locks_released=locks,
        all_checks_and_viewed_image_hashes=True,failed100_updates_preserved=True,
        matched1200_updates_and_logs_receipts=True,no_prior_order_sign_or_support_changes=True,
        both_main_plans_task_section_committed=True,results_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        source_sha256=sha(Path(__file__))))
    print('L8R_FINAL_VALIDATION_PASS',len(frozen),len(committed),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--enable-final-validation',action='store_true')
    if not p.parse_args().enable_final_validation:raise SystemExit('Default off')
    run()
