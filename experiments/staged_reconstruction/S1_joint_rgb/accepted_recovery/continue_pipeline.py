"""Explicit one-shot serial train/evaluate/export orchestrator; manual review stays pending."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from experiments.staged_reconstruction.S1_joint_rgb.accepted_recovery.launch import configure, BASE, REPORT, RUNS, command
from variants.staged_reconstruction.prepare_data import read, sha, ROOT


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-accepted-joint-runner', action='store_true')
    args = parser.parse_args()
    assert args.enable_accepted_joint_runner
    io = configure(); io.verify(); assert not io.stopped()
    assert read(REPORT/'recovery.json')['passed']
    with io.lock('orchestrator.lock'):
        result = {'status': 'running', 'pid': os.getpid(), 'source_sha256': sha(Path(__file__))}
        io.atomic_json(RUNS/'orchestrator.json', result)
        try:
            result['train_evaluation'] = command(io, 'pipeline')
            if io.stopped():
                result['status'] = 'saved_stopped'; io.atomic_json(RUNS/'orchestrator.json', result); return
            if not (REPORT/'exports.json').exists():
                log = REPORT/'native_exports.log'
                with log.open('x') as stream:
                    code = subprocess.run([sys.executable, '-u', str(BASE/'export.py')], cwd=ROOT,
                        stdout=stream, stderr=subprocess.STDOUT).returncode
                assert code == 0, 'Preserve failed export log '+str(log)
                result['export_log_sha256'] = sha(log)
            result['status'] = 'evaluated_exported_pending_manual_review'
            io.atomic_json(RUNS/'orchestrator.json', result)
        except Exception as error:
            result.update(status='failed', error=repr(error)); io.atomic_json(RUNS/'orchestrator.json', result); raise


if __name__ == '__main__': main()
