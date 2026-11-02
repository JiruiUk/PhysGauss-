"""Wait for the existing GPU queue, then run the three analysis stages once."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'experiments/flow_comparison'
STATUS = REPORTS/'postprocess_status.json'

def record(status, stage=None, error=None):
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps({'status': status, 'stage': stage, 'error': error,
        'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2)+'\n', encoding='utf-8')
    temporary.replace(STATUS)

try:
    record('waiting_for_training_and_exports')
    deadline = time.monotonic()+3600
    while True:
        progress = json.loads((REPORTS/'progress.json').read_text(encoding='utf-8-sig'))
        if progress['status'] == 'completed':
            break
        if progress['status'] == 'failed':
            raise RuntimeError(f"Training/render queue failed: {progress.get('error')}")
        if time.monotonic()>deadline:
            raise TimeoutError('Existing training/render queue did not complete within an hour')
        time.sleep(10)
    for stage in ['evaluate_motion', 'gradient_balance', 'finalize']:
        record('running', stage)
        with (REPORTS/(stage+'.log')).open('wb') as stream:
            result = subprocess.run([sys.executable, '-u', str(Path(__file__).parent/(stage+'.py'))],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{stage} exited {result.returncode}; see its log')
    record('completed')
except Exception as error:
    record('failed', error=repr(error))
    raise
