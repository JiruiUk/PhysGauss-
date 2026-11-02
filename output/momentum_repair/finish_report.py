"""Resume only postprocessing; preserve completed models and original lock."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
import torch
torch.set_num_threads(4)
from variants.momentum_guard.run import NAME,RUNS,REPORTS,assert_frozen,monitor
from variants.metric_tuning.run import read_json,write_json,sha256,now
run=RUNS/NAME/'rubber_duck';out=ROOT/'output/momentum_repair'/NAME
assert_frozen(ROOT/'variants/momentum_guard/code')
checkpoint=run/'checkpoints/resume.pt';meta=read_json(checkpoint.with_suffix('.json'))
assert meta['iteration']==50000 and sha256(checkpoint)==meta['sha256']
lock=RUNS/'selection_lock.json';original_lock=sha256(lock)
assert 'ours_best' in read_json(run/'results.json') and 'ours_50000' in read_json(run/'results.json')
# Pre-create directories serially before the inherited parallel image exporter.
for tag in ['best','50000']:(out/'pairs'/tag).mkdir(parents=True,exist_ok=True)
try:
    from variants.momentum_guard.export import export,report
    monitor('exporting');export(run)
    assert sha256(lock)==original_lock
    monitor('diagnosing');report()
    assert sha256(lock)==original_lock
    assert_frozen(ROOT/'variants/momentum_guard/code');monitor('completed')
    write_json(REPORTS/'postprocess_recovery.json',{'status':'complete','completed_utc':now(),
        'training_or_rendering_repeated':False,'selection_lock_sha256_preserved':original_lock,
        'complete_checkpoint_sha256':meta['sha256'],
        'repair':'Serialize creation of pair directories before inherited parallel exporter; decode live status as UTF-8. Frozen training sources untouched.'})
except Exception as error:monitor('failed',repr(error));raise
