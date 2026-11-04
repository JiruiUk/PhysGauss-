"""Verify native exact-support evidence and pinned sources, not geometry truth."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-native120', action='store_true')
    args = parser.parse_args()
    exp = ROOT/'experiments/staged_reconstruction/uncertainty_motion'
    out = ROOT/'output/staged_reconstruction/uncertainty_motion'
    for manifest in ['official_source.json', 'backend_source.json']:
        for row in read(exp/manifest)['downloaded_audit_text']:
            path = ROOT/row['path']; value = path.read_bytes()
            assert sha(path) == row['sha256']
            assert hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest() == row['official_blob_sha']
    tiny = read(exp/'qualification_protocol_v4_native_audit.json')
    protocol = read(exp/'qualification_protocol_v4.json')
    assert tiny['qualification_protocol_sha256'] == sha(exp/'qualification_protocol_v4.json')
    for name, expected in protocol['sources'].items():
        assert sha(ROOT/name) == expected, name
    weights = np.load(out/'tiny_exact_weights.npz')
    assert sha(out/'tiny_exact_weights.npz') == tiny['explicit_weights_sha256']
    exact = np.square(weights['exact_color_weights']).sum(1)
    np.testing.assert_allclose(exact, weights['squared_support'], rtol=1e-14, atol=1e-12)
    discrepancy = np.abs(exact-weights['truncated_export_support']).sum()/exact.sum()
    assert abs(discrepancy-tiny['export_buffer_relative_L1_error']) < 1e-12
    assert (weights['exact_color_weights'][20:32].sum(1) > 0).all()
    assert exact[-1] == 0
    assert all(v['passed'] for v in tiny['independent_seeds'] if v['batches'] == 512)
    assert all(not v['passed'] for v in tiny['independent_seeds'] if v['batches'] == 128)
    results = []
    names = ['observed_pilot']+(['observed_native120'] if args.include_native120 else [])
    for name in names:
        plan_path = exp/(name+'_protocol.json'); plan = read(plan_path)
        result = read(exp/name/'result.json')
        assert result['protocol_sha256'] == sha(plan_path)
        assert result['all_model_tensors_and_RNG_unchanged'] and result['optimizer_updates'] == 0
        for group in ['sources', 'inputs']:
            for filename, expected in plan[group].items():
                assert sha(ROOT/filename) == expected, filename
        assert [[r['frame'], r['width']] for r in result['cases']] == plan['cases']
        sample_errors = []
        for row in result['cases']:
            basename = f't{row["frame"]:04d}_w{row["width"]}'
            array_path = out/name/(basename+'.npz')
            assert sha(array_path) == row['array_sha256']
            assert sha(out/name/(basename+'.png')) == row['figure_sha256']
            a = np.load(array_path)
            assert a['support'].shape == (2, 15507)
            assert all(np.isfinite(a[k]).all() for k in a.files)
            ids, exact = a['exact_sample_rows'], a['exact_support']
            assert len(ids) == plan['exact_sample_rows']
            nonzero = exact > 1e-6
            errors = np.abs(a['support'][:, ids][:, nonzero]-exact[nonzero])/exact[nonzero]
            median, p90 = np.median(errors, axis=1), np.quantile(errors, .9, axis=1)
            np.testing.assert_allclose(median, row['exact_onehot_checks']['median_relative_error_per_seed'], rtol=1e-12)
            np.testing.assert_allclose(p90, row['exact_onehot_checks']['p90_relative_error_per_seed'], rtol=1e-12)
            assert nonzero.sum() >= plan['exact_sample_active_min']
            assert (median <= plan['exact_relative_median_max']).all()
            assert (p90 <= plan['exact_relative_p90_max']).all()
            assert row['native_flags_off_RGB_alpha_exact']
            sample_errors.extend(median.tolist())
        assert result['passed'] and all(r['passed'] for r in result['cases'])
        results.append({'name': name, 'cases': len(result['cases']), 'passed': True,
            'exact_support_error_median_across_cases_and_seeds': float(np.median(sample_errors)),
            'worst_case_median_relative_error': max(sample_errors), 'result_sha256': sha(exp/name/'result.json')})
    receipt = {'passed': True, 'official_and_backend_blob_hashes_verified': True,
        'tiny_exact_weights_and_truncated_export_error_recomputed': True, 'observed_audits': results,
        'script_sha256': sha(__file__), 'test_RGB_read': False, 'optimizer_updates': 0,
        'scope': 'Verifies pinned numerical evidence; does not certify object completeness or physical identity'}
    path = exp/('verification_native120.json' if args.include_native120 else 'verification_pilot.json')
    assert not path.exists(), 'Preserve independent verification receipts'
    write(path, receipt)
    print('COLOR_SUPPORT_EVIDENCE_VERIFIED', receipt, flush=True)


if __name__ == '__main__':
    main()
