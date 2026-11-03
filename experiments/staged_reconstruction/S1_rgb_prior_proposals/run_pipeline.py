"""Serialize one frozen proposal, full saved-output audit and independent GT."""
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.rgb_prior_proposals.prepare import REPORT, RUNS, SOURCE, verify, stopped


def main():
    assert not stopped(); verify(); RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS/'pipeline.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); stage = 'inference'
    def run(label, module, args):
        assert not stopped()
        log = REPORT/f'{attempt}_{label}.log'
        with log.open('wb') as output:
            child = subprocess.Popen([sys.executable, '-u', '-m', module, *args], cwd=ROOT, stdout=output,
                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            write(RUNS/'pipeline.json', {'status': 'running', 'stage': label, 'pid': os.getpid(), 'child_pid': child.pid,
                'attempt': attempt, 'log': log.relative_to(ROOT).as_posix(), 'runner_sha256': sha(Path(__file__))})
            assert child.wait() == 0, f'{label} failed; preserved {log}'
    try:
        if not (REPORT/'inference.json').exists():
            run('inference', 'variants.staged_reconstruction.rgb_prior_proposals.infer', ['--enable-rgb-prior-proposals'])
        stage = 'input_audit'
        if not (REPORT/'input_audit.json').exists(): run('input_audit', 'variants.staged_reconstruction.rgb_prior_proposals.audit', [])
        stage = 'GT_integrity'
        for path, h in read(SOURCE/'dataset.json')['files'].items(): assert sha(ROOT/path) == h, path
        write(REPORT/f'{attempt}_evaluation_integrity.json', {'passed': True, 'all_train_and_GT_file_hashes_verified': True,
            'dataset_receipt_sha256': sha(SOURCE/'dataset.json')})
        stage = 'evaluation'
        if not (REPORT/'evaluation.json').exists(): run('evaluation', 'variants.staged_reconstruction.rgb_prior_proposals.evaluate', [])
        verify(); write(RUNS/'pipeline.json', {'status': 'evaluated_pending_review', 'attempt': attempt,
            'Gaussian_training_started': False, 'real_training_authorized': False, 'runner_sha256': sha(Path(__file__))})
    except Exception as error:
        write(RUNS/'pipeline.json', {'status': 'saved_stopped' if stopped() else 'failed', 'stage': stage, 'error': str(error), 'attempt': attempt})
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__': main()
