"""Finalize an actually viewed D7 measurement without changing inference."""
from datetime import datetime, timezone
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.plane_transport.prepare import BASE, REPORT, RUNS, verify


def main():
    verify()
    assert not (REPORT/'final_review.json').exists()
    assert read(REPORT/'input_audit.json')['passed']
    assert read(REPORT/'preflight.json')['passed']
    gate = read(REPORT/'gate.json')
    assert not gate['both_cases_numeric_improvement'] and not gate['both_cases_Gaussian_evidence']
    exports = read(REPORT/'exports.json')
    assert exports['original_GT_receipt_matches_preexisting_D5_freeze']
    assert exports['all192_independent_aggregate_replay_passed']
    for name, h in exports['images'].items():
        assert sha(ROOT/name) == h
    pipeline = read(RUNS/'pipeline.json')
    assert pipeline['status'] == 'measured_pending_review'
    assert [row['stage'] for row in pipeline['history']] == ['inference', 'input_audit', 'evaluation']
    for row in pipeline['history']:
        assert row['exit_code'] == 0 and sha(ROOT/row['log']) == row['log_sha256']
    evidence = ['source_freeze.json', 'tests.log', 'preflight.json', 'analytic_controls.json',
        'inference.json', 'input_audit.json', 'evaluation.json', 'gate.json', 'exports.json',
        'coverage_diagnostic.json', 'assessment.md']
    review = {'status': 'complete_rejected_geometry_gate', 'pending_review': False,
        'manual_review_completed': True, 'reviewed_images': exports['images'],
        'actually_viewed': 'Both cases fixed0/24/72 RGB, same-pixel D6/D7 errors and change maps; all A/B/C analytic control depths.',
        'all192_membership_source_query_plane_acceptance_and_hash_replay': True,
        'fixed0_24_72_full_raw_RGB_replay_exact': True,
        'independent_GT_count_MAE_P90_replay_and_original_dataset_hashes_passed': True,
        'flags_off_identity_and_two_condition_three_process4_vs2plus2_recovery_passed': True,
        'gates_unchanged': True, 'Gaussian_or_real_training_started': False,
        'native86_188_PLY_video_or_Adam_recovery_claim': False,
        'S3U_and_unknown_identity_preserved': True,
        'posthoc_coverage_diagnostic_no_GT_feedback': True,
        'next': 'Preregister full-RGB no-update static-revocation audit J0; reuse existing parallel coupling evidence, no depth-training promotion.',
        'next_plan_sha256': sha(BASE/'next_plan.md'),
        'evidence_sha256': {name: sha(REPORT/name) for name in evidence},
        'reviewed_at_utc': datetime.now(timezone.utc).isoformat(), 'finalizer_sha256': sha(Path(__file__))}
    write(REPORT/'final_review.json', review)
    status = {'status': review['status'], 'pending_review': False, 'Gaussian_training': False,
        'final_review_sha256': sha(REPORT/'final_review.json')}
    write(RUNS/'final_status.json', status)
    write(RUNS/'state.json', status)
    pipeline.update(status='complete', stage='complete', result=review['status'], pending_review=False,
        final_review_sha256=sha(REPORT/'final_review.json'))
    write(RUNS/'pipeline.json', pipeline)
    print('D7_FINAL_REVIEW_COMPLETE_REJECTED_GEOMETRY_GATE', flush=True)


if __name__ == '__main__':
    main()
