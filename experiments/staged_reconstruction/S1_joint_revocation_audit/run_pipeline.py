"""One locked full J0 measurement after exact native no-update reload checks."""
import argparse
from datetime import datetime, timezone
import msvcrt
import os
import subprocess
import sys
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT, read, sha
from variants.staged_reconstruction.rgb_prior_proposals.io import atomic_json
from variants.staged_reconstruction.joint_revocation_audit.prepare import BASE, REPORT, RUNS, verify, stopped


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-joint-revocation-pipeline', action='store_true')
    args = parser.parse_args()
    assert args.enable_joint_revocation_pipeline and not stopped()
    verify(); assert read(REPORT/'preflight.json')['passed']
    for name, h in read(REPORT/'repair_source_freeze.json').items():
        assert sha(ROOT/name) == h
    lock = (RUNS/'pipeline.lock').open('a+b'); lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    state = {'status': 'running', 'stage': 'six_frame_measurement', 'pid': os.getpid(),
        'runner_sha256': sha(Path(__file__)), 'optimizer_updates': 0}
    try:
        assert not (REPORT/'formal.json').exists(), 'Do not repeat completed J0 measurements'
        atomic_json(RUNS/'pipeline.json', state)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log = REPORT/f'{stamp}_formal.log'
        command = [sys.executable, '-u', str(BASE/'run_worker_receipt_repair.py'),
            '--enable-joint-revocation-audit', '--phase', 'formal']
        with log.open('x') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        state.update(command=command, exit_code=result.returncode,
            log=log.relative_to(ROOT).as_posix(), log_sha256=sha(log))
        assert result.returncode == 0 and (REPORT/'formal.json').exists(), 'Preserve failed log '+str(log)
        state.update(status='measured_pending_review', stage='manual_review')
        atomic_json(RUNS/'pipeline.json', state)
        print('J0_PIPELINE_MEASURED_PENDING_REVIEW', flush=True)
    except Exception as error:
        state.update(status='saved_stopped' if stopped() else 'failed', error=repr(error))
        atomic_json(RUNS/'pipeline.json', state)
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
