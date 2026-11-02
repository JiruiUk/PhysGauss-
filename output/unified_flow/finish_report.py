"""Finish existing exports without re-entering any training or rendering path.

The coordinator imports the extended architecture before the diagnostic sets
its hash-encoder paths. Set those paths before any architecture import here.
Frozen training modules and the unrelated stale root build lock stay intact.
"""
import faulthandler
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['PIDG_HASH_SOURCE_DIR'] = str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR'] = str(ROOT/'third_party/PIDG/tmp_build')
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)

import torch
torch.set_num_threads(4)
from variants.unified_flow.run import assert_frozen, monitor, NAMES, RUNS

assert_frozen(ROOT/'variants/unified_flow/code')
for name in NAMES:
    receipt = ROOT/'experiments/unified_flow'/f'{name}_exports.json'
    assert receipt.exists(), receipt
    state = json.loads((RUNS/name/'rubber_duck/checkpoints/resume.json').read_text())
    assert state['iteration'] == 50000
print('Frozen inputs and both complete runs verified; diagnostic import paths set.', flush=True)
monitor('diagnosing')
try:
    from variants.unified_flow.export import report
    report()
    assert_frozen(ROOT/'variants/unified_flow/code')
    monitor('completed')
    print('Report completed '+datetime.now(timezone.utc).isoformat(), flush=True)
except Exception as error:
    monitor('failed', error=repr(error))
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
