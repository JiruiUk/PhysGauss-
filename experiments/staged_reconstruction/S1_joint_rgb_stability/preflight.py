"""Single fixed numerical control; stop on first failure without a retry sweep."""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from variants.staged_reconstruction.prepare_data import ROOT, read, sha
from variants.staged_reconstruction.joint_rgb.core import assert_equivalent
from variants.staged_reconstruction.joint_rgb_stability.adapter import configure


def main():
    io = configure(); protocol = read(io.BASE/'protocol.json')
    assert not io.stopped() and not (io.REPORT/'recovery.json').exists()
    assert not (io.REPORT/'recovery_failure.json').exists(), 'No automatic same-control retry'
    io.verify(); assert read(io.REPORT/'native_check.json')['passed']
    torch.set_num_threads(4); rows = []
    with io.lock('preflight.lock'):
        try:
            for case in protocol['cases']:
                for arm in protocol['arms']:
                    commands = []
                    for tag, stop in [('full', None), ('split', protocol['recovery_split_steps']), ('split', None)]:
                        assert not io.stopped()
                        command = [sys.executable, '-u', str(io.BASE/'run_worker.py'),
                            '--enable-joint-rgb-stability', '--case', case, '--arm', arm, '--preflight-tag', tag]
                        if stop:
                            command += ['--stop-after', str(stop)]
                        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                        log = io.REPORT/f'{stamp}_{case}_{arm}_{tag}.log'
                        with log.open('x') as stream:
                            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                        commands.append({'command': command, 'log': log.relative_to(ROOT).as_posix(), 'sha256': sha(log)})
                        assert result.returncode == 0, str(log)
                        if stop:
                            split = io.RUNS/'preflight/split'/case/arm
                            (split/'split20.pt').write_bytes((split/'checkpoint.pt').read_bytes())
                    states = [torch.load(io.RUNS/'preflight'/tag/case/arm/'checkpoint.pt', map_location='cpu', weights_only=False)
                        for tag in ['full', 'split']]
                    a, b = states
                    assert a['iteration'] == b['iteration'] == protocol['recovery_full_steps']
                    assert_equivalent(a, b, protocol['recovery_atol'], protocol['recovery_rtol'])
                    parent = torch.load(ROOT/protocol['parents'][case]['checkpoint'], map_location='cpu', weights_only=False)
                    if arm == 'frozen_static':
                        for name, value in parent['model'].items():
                            assert torch.equal(a['model']['static.'+name], value)
                        for group in parent['optimizer']['param_groups']:
                            group['eps'] = 1e-8
                        assert_equivalent(a['optimizers'][0], parent['optimizer'], 0, 0)
                    else:
                        for name in ['_xyz', '_scaling', '_rotation']:
                            assert torch.equal(a['model']['static.'+name], parent['model'][name])
                    rows.append({'case': case, 'arm': arm, 'passed': True, 'commands': commands})
                    print('J1_R1_GROUP_RECOVERY_PASSED', case, arm, flush=True)
            io.verify()
            io.atomic_json(io.REPORT/'recovery.json', {'passed': True, 'rows': rows,
                'common_Adam_eps': 1e-8, 'atol': protocol['recovery_atol'], 'rtol': protocol['recovery_rtol'],
                'original_J1_failure_retained': True, 'formal_training_admitted': False, 'source_sha256': sha(Path(__file__))})
            io.atomic_json(io.RUNS/'state.json', {'status': 'numerical_preflight_passed', 'formal_training_admitted': False})
        except Exception as error:
            result = {'passed': False, 'completed_groups': rows, 'failed_case': case, 'failed_arm': arm,
                'commands': commands, 'error': repr(error), 'original_tolerance_unchanged': True,
                'formal_training_admitted': False, 'single_control_no_sweep': True, 'source_sha256': sha(Path(__file__))}
            io.atomic_json(io.REPORT/'recovery_failure.json', result)
            io.atomic_json(io.RUNS/'state.json', {'status': 'numerical_preflight_failed', **result})
            raise


if __name__ == '__main__':
    main()
