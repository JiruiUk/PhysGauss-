"""One serial four-arm Gaussian run; leaves final independent evaluation explicit."""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from variants.staged_reconstruction.prepare_data import ROOT, read, sha
from variants.staged_reconstruction.joint_rgb.io import BASE, REPORT, RUNS, atomic_json, stopped, verify, lock


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-joint-rgb-pipeline', action='store_true')
    args = parser.parse_args()
    assert args.enable_joint_rgb_pipeline and not stopped()
    verify(); assert read(REPORT/'recovery.json')['passed']
    assert not (RUNS/'final_status.json').exists()
    with lock('pipeline.lock'):
        state = {'status': 'running', 'pid': os.getpid(), 'completed': [], 'source_sha256': sha(__file__)}
        atomic_json(RUNS/'pipeline.json', state)
        protocol = read(BASE/'protocol.json')
        try:
            for case in protocol['cases']:
                for arm in protocol['arms']:
                    if stopped():
                        state.update(status='saved_stopped'); atomic_json(RUNS/'pipeline.json', state); return
                    directory = RUNS/'formal'/case/arm
                    if (directory/'checkpoint.json').exists() and read(directory/'checkpoint.json')['iteration'] == protocol['updates']:
                        assert sha(directory/'checkpoint.pt') == read(directory/'checkpoint.json')['sha256']
                        state['completed'].append({'case': case, 'arm': arm, 'already_completed': True}); continue
                    state.update(case=case, arm=arm); atomic_json(RUNS/'pipeline.json', state)
                    command = [sys.executable, '-u', '-m', 'variants.staged_reconstruction.joint_rgb.train',
                        '--enable-joint-rgb', '--case', case, '--arm', arm]
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                    log = REPORT/f'{stamp}_formal_{case}_{arm}.log'
                    with log.open('x') as stream:
                        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    if stopped():
                        state.update(status='saved_stopped'); atomic_json(RUNS/'pipeline.json', state); return
                    assert result.returncode == 0, 'Preserve failed log '+str(log)
                    assert read(directory/'checkpoint.json')['iteration'] == protocol['updates']
                    state['completed'].append({'case': case, 'arm': arm, 'log': log.relative_to(ROOT).as_posix(), 'log_sha256': sha(log)})
                    atomic_json(RUNS/'pipeline.json', state)
            state.update(status='trained_pending_independent_evaluation', stage='independent_evaluation_required')
            atomic_json(RUNS/'pipeline.json', state)
            print('J1_FOUR_ARMS_TRAINED_PENDING_INDEPENDENT_EVALUATION', flush=True)
        except Exception as error:
            state.update(status='failed', error=repr(error)); atomic_json(RUNS/'pipeline.json', state); raise


if __name__ == '__main__':
    main()
