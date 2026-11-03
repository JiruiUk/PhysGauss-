"""Finalize the read-only milestone after actual image review; retain raw receipts."""
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from variants.staged_reconstruction.prepare_data import ROOT, read, sha
from variants.staged_reconstruction.joint_rgb.io import atomic_json


def main():
    base = Path(__file__).resolve().parent; report = base/'v1'
    runs = ROOT/'runs/pidg_staged_reconstruction/S1_joint_attribution/v1'
    assert read(runs/'state.json')['status'] == 'measured_pending_manual_review'
    assert not (report/'final_review.json').exists()
    measured = read(report/'measurement.json'); audit = read(report/'independent_audit.json')
    assert audit['passed'] and audit['measurement_sha256'] == sha(report/'measurement.json')
    assert read(report/'native_replay.json')['passed'] and read(report/'unit_tests.json')['passed']
    for name, expected in read(report/'source_freeze.json').items(): assert sha(ROOT/name) == expected, name
    expected_names = {f'output/staged_reconstruction/S1_joint_attribution/v1/{case}_c{cam:02d}_{kind}.png'
        for case in ['diffuse', 'glossy'] for cam in [0, 8, 24] for kind in ['opacity_SH', 'ray_changes']}
    assert expected_names == measured['images'].keys()
    for name, expected in measured['images'].items():
        assert sha(ROOT/name) == expected
        with Image.open(ROOT/name) as image: image.load()
    manual = {'actually_viewed_all12_fixed_native_figures': True, 'images': measured['images'],
        'findings': ['c8/c24 RGB blocks fade under either opacity or SH changes',
            'alpha retains floating support and base halos',
            'lost-valid rays on pillar/base dominate missing rays; gains include thin structure',
            'fixed intervention not training causal attribution or physical identity'],
        'independent_GT_appears_only_in_evaluation': True}
    atomic_json(report/'manual_review.json', manual)
    result = {'status': 'completed_read_only_attribution', 'pending_review': False,
        'training_updates': 0, 'J1_A_status': 'completed_not_promoted',
        'Rubber_Duck_training_admitted': False, 'quality_gates_changed': False,
        'measurement_sha256': sha(report/'measurement.json'), 'independent_audit_sha256': sha(report/'independent_audit.json'),
        'manual_review_sha256': sha(report/'manual_review.json'), 'assessment_sha256': sha(report/'assessment.md'),
        'completed_UTC': datetime.now(timezone.utc).isoformat()}
    atomic_json(report/'final_review.json', result)
    for filename in ['state.json', 'final_status.json']: atomic_json(runs/filename, result)
    print('J2_FINAL_REVIEW_COMPLETE', flush=True)


if __name__ == '__main__': main()
