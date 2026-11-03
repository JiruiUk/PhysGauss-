"""Serialized locked D7 inference, input replay and independent GT evaluation."""
import argparse
from datetime import datetime, timezone
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
from variants.staged_reconstruction.prepare_data import ROOT, read, sha
from variants.staged_reconstruction.rgb_prior_proposals.io import atomic_json
from variants.staged_reconstruction.plane_transport.prepare import REPORT, RUNS, verify, stopped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-plane-transport-pipeline', action='store_true')
    args = parser.parse_args()
    assert args.enable_plane_transport_pipeline and not stopped()
    verify()
    assert read(REPORT/'preflight.json')['passed']
    lock = (RUNS/'pipeline.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    path = RUNS/'pipeline.json'
    state = read(path) if path.exists() else {'status': 'starting', 'history': []}
    assert state['status'] not in ('complete', 'measured_pending_review'), 'Do not repeat completed measurements'
    state['runner_sha256'] = sha(Path(__file__))
    state['pid'] = os.getpid()
    try:
        for stage, module, extra, receipt in [
            ('inference', 'infer', ['--enable-plane-transport'], 'inference.json'),
            ('input_audit', 'audit', [], 'input_audit.json'),
            ('evaluation', 'evaluate', [], 'evaluation.json')]:
            assert not stopped(), 'User stop pending; do not launch a new stage'
            verify()
            if (REPORT/receipt).exists():
                if receipt == 'input_audit.json':
                    assert read(REPORT/receipt)['passed']
                continue
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            log = REPORT/f'{stamp}_{stage}.log'
            command = [sys.executable, '-u', '-m', 'variants.staged_reconstruction.plane_transport.'+module]+extra
            state.update(status='running', stage=stage)
            atomic_json(path, state)
            print('D7_STAGE_START', stage, str(log), flush=True)
            with log.open('x') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            state['history'].append({'stage': stage, 'command': command, 'exit_code': result.returncode,
                'log': log.relative_to(ROOT).as_posix(), 'log_sha256': sha(log)})
            assert result.returncode == 0, stage+' failed; preserve log '+str(log)
            assert (REPORT/receipt).exists()
            atomic_json(path, state)
            print('D7_STAGE_COMPLETE', stage, flush=True)
        verify()
        state.update(status='measured_pending_review', stage='manual_review')
        atomic_json(path, state)
        atomic_json(RUNS/'state.json', {'status': 'evaluated_pending_manual_review', 'Gaussian_training': False})
        print('D7_PIPELINE_MEASURED_PENDING_REVIEW', flush=True)
    except Exception as error:
        state.update(status='saved_stopped' if stopped() else 'failed', error=repr(error))
        atomic_json(path, state)
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
