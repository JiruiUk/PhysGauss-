"""Independent full40 versus20+20 processes, complete state and camera recovery."""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from variants.staged_reconstruction.prepare_data import ROOT, read, sha
from variants.staged_reconstruction.joint_rgb.core import assert_equivalent
from variants.staged_reconstruction.joint_rgb.io import BASE, REPORT, RUNS, atomic_json, verify, stopped, lock


def worker(case, arm, tag, stop=None):
    command = [sys.executable, '-u', '-m', 'variants.staged_reconstruction.joint_rgb.train',
        '--enable-joint-rgb', '--case', case, '--arm', arm, '--preflight-tag', tag]
    if stop:
        command += ['--stop-after', str(stop)]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log = REPORT/f'{stamp}_{case}_{arm}_{tag}.log'
    with log.open('x') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    assert result.returncode == 0, 'Preserve failed log '+str(log)
    return {'command': command, 'log': log.relative_to(ROOT).as_posix(), 'sha256': sha(log)}


def main():
    assert not stopped() and not (REPORT/'recovery.json').exists()
    verify(); protocol = read(BASE/'protocol.json'); rows = []
    with lock('preflight.lock'):
        for case in protocol['cases']:
            for arm in protocol['arms']:
                assert not (RUNS/'preflight/full'/case/arm/'checkpoint.pt').exists(), 'Do not overwrite preflight'
                commands = [worker(case, arm, 'full'), worker(case, arm, 'split', protocol['recovery_split_steps'])]
                split = RUNS/'preflight/split'/case/arm
                middle = torch.load(split/'checkpoint.pt', map_location='cpu', weights_only=False)
                assert middle['iteration'] == protocol['recovery_split_steps']
                (split/'split20.pt').write_bytes((split/'checkpoint.pt').read_bytes())
                commands.append(worker(case, arm, 'split'))
                a = torch.load(RUNS/'preflight/full'/case/arm/'checkpoint.pt', map_location='cpu', weights_only=False)
                b = torch.load(split/'checkpoint.pt', map_location='cpu', weights_only=False)
                assert a['iteration'] == b['iteration'] == protocol['recovery_full_steps']
                assert_equivalent(a, b, protocol['recovery_atol'], protocol['recovery_rtol'])
                parent = torch.load(ROOT/protocol['parents'][case]['checkpoint'], map_location='cpu', weights_only=False)
                for name in ['_xyz', '_scaling', '_rotation']:
                    assert torch.equal(a['model']['static.'+name], parent['model'][name])
                if arm == 'frozen_static':
                    for name, value in parent['model'].items():
                        assert torch.equal(a['model']['static.'+name], value), name
                    assert_equivalent(a['optimizers'][0], parent['optimizer'], atol=0, rtol=0)
                else:
                    assert not torch.equal(a['model']['static._opacity'], parent['model']['_opacity'])
                rows.append({'case': case, 'arm': arm, 'passed': True, 'commands': commands,
                    'full40_sha256': sha(RUNS/'preflight/full'/case/arm/'checkpoint.pt'),
                    'split20_sha256': sha(split/'split20.pt'), 'split40_sha256': sha(split/'checkpoint.pt')})
            control = torch.load(RUNS/'preflight/full'/case/'frozen_static/checkpoint.pt', map_location='cpu', weights_only=False)
            candidate = torch.load(RUNS/'preflight/full'/case/'revisable_static/checkpoint.pt', map_location='cpu', weights_only=False)
            assert control['sample_trace'] == candidate['sample_trace']
            assert_equivalent(control['camera_stack'], candidate['camera_stack'])
        verify()
        atomic_json(REPORT/'recovery.json', {'passed': True, 'rows': rows,
            'full40_vs_independent20_plus20_model_both_Adam_all_RNG_camera_stack_visits_trace': True,
            'static_geometry_fixed_and_control_moments_unchanged': True,
            'matched_control_treatment_camera_sequence': True,
            'atol': protocol['recovery_atol'], 'rtol': protocol['recovery_rtol'],
            'source_manifest_sha256': sha(REPORT/'source_freeze.json'), 'source_sha256': sha(Path(__file__))})
        print('J1_COMPLETE_INDEPENDENT_RECOVERY_PASSED', flush=True)


if __name__ == '__main__':
    main()
