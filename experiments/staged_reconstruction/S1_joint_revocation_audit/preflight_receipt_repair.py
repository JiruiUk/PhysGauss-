"""Independent preflight orchestrator for the unchanged J0 calculation worker."""
import argparse
from datetime import datetime, timezone
import subprocess
import sys
import numpy as np
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.joint_revocation_audit.prepare import BASE, REPORT, RUNS, stopped, verify


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-joint-revocation-preflight', action='store_true')
    args = parser.parse_args()
    assert args.enable_joint_revocation_preflight and not stopped()
    assert not (REPORT/'preflight.json').exists()
    for name, h in read(REPORT/'repair_source_freeze.json').items():
        assert sha(ROOT/name) == h
    commands = []
    for number in ['0', '1']:
        assert not stopped(); verify()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log = REPORT/f'{stamp}_receipt_repair_preflight_{number}.log'
        command = [sys.executable, '-u', str(BASE/'run_worker_receipt_repair.py'),
            '--enable-joint-revocation-audit', '--phase', 'preflight', '--replicate', number]
        with log.open('x') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        assert result.returncode == 0, 'Preserve failed preflight '+str(log)
        commands.append({'command': command, 'log': log.relative_to(ROOT).as_posix(), 'sha256': sha(log)})
    a, b = read(REPORT/'preflight_0.json'), read(REPORT/'preflight_1.json')
    assert a['model_hash_before_and_after'] == b['model_hash_before_and_after']
    assert a['imported_sources'] == b['imported_sources']
    plan = read(BASE/'protocol.json')
    for filename in ['gradient_render.npz', 'native_render.npz']:
        with np.load(RUNS/'preflight_0'/filename) as first, np.load(RUNS/'preflight_1'/filename) as second:
            assert first.files == second.files
            for key in first.files:
                assert np.array_equal(first[key], second[key]), filename+'/'+key
    for key, left in a['rows'][0]['gradient_groups'].items():
        right = b['rows'][0]['gradient_groups'][key]
        for field in left:
            if field in ['l2', 'mean_abs']:
                assert np.isclose(left[field], right[field], atol=plan['independent_reload_gradient_atol'],
                    rtol=plan['independent_reload_gradient_rtol']), key+'/'+field
            else:
                assert left[field] == right[field], key+'/'+field
    assert a['rows'][0]['full_RGB_loss'] == b['rows'][0]['full_RGB_loss']
    write(REPORT/'preflight.json', {'passed': True, 'separate_processes': 2,
        'same_pinned_model_parameters_and_native_800_1600_RGB_alpha_exact': True,
        'finite_nonzero_static_and_dynamic_gradients_and_flags_off_parity': True,
        'aggregate_gradient_tolerances_unchanged': True, 'caller_RNG_flags_and_values_preserved': True,
        'imported_sources': a['imported_sources'], 'commands': commands, 'optimizer_updates': 0,
        'Adam_RNG_camera_training_recovery_claim': False,
        'source_sha256': sha(REPORT/'source_freeze.json'),
        'evaluation_receipt_repair_source_sha256': sha(REPORT/'repair_source_freeze.json')})
    print('J0_PREFLIGHT_PASSED_WITH_INDEPENDENT_RECEIPT_REPAIR', flush=True)


if __name__ == '__main__':
    main()
