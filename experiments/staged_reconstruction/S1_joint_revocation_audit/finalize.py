"""Finalize an already measured, independently checked, manually viewed audit."""
from datetime import datetime, timezone
from pathlib import Path
import msvcrt
from variants.staged_reconstruction.prepare_data import ROOT, read, sha, write

BASE = ROOT/'experiments/staged_reconstruction/S1_joint_revocation_audit'
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_joint_revocation_audit/v1'
OUT = ROOT/'output/staged_reconstruction/S1_joint_revocation_audit/v1'


def main():
    assert not (REPORT/'final_review.json').exists(), 'Do not repeat final review'
    locks = []
    try:
        for name in ['pipeline.lock', 'audit.lock']:
            stream = (RUNS/name).open('r+b'); stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1); locks.append(stream)
        protocol = read(BASE/'protocol.json')
        review = read(REPORT/'independent_review.json')
        formal = read(REPORT/'formal.json')
        assert review['passed'] and formal['optimizer_updates'] == 0
        assert review['formal_sha256'] == sha(REPORT/'formal.json')
        assert review['source_sha256'] == sha(BASE/'export_review.py')
        assert sha(ROOT/protocol['checkpoint']) == protocol['checkpoint_sha256']
        for receipt in ['source_freeze.json', 'repair_source_freeze.json']:
            for name, expected in read(REPORT/receipt).items():
                assert sha(ROOT/name) == expected, name
        for name, expected in {**formal['imported_sources'], **formal['images'], **review['review_images']}.items():
            assert sha(ROOT/name) == expected, name
        viewed = ['fixed_old_model_review.png', 'observed_old_model_review.png',
            'static_opacity_gradient_signs.png', 'fixed86_t0100_2_native.png',
            'fixed188_t0200_2_native.png']
        result = {'status': 'completed_no_update_audit', 'pending_review': False,
            'completed_at_UTC': datetime.now(timezone.utc).isoformat(),
            'implementation_commit': 'b7e304ba', 'measurement_commit': 'c9ebd762',
            'optimizer_updates': 0, 'gradient_path_qualified': True,
            'cleanup_or_geometry_proven': False, 'new_Gaussian_training': False,
            'Adam_RNG_camera_training_recovery_claim': False,
            'legacy_conditional_color_mask_training_disclosed': True,
            'no_hybrid_static_bank': True, 'D7_training_promotion': False,
            'Rubber_Duck_continuation_admitted': False,
            'independent_review_sha256': sha(REPORT/'independent_review.json'),
            'assessment_sha256': sha(REPORT/'assessment.md'),
            'checkpoint_sha256': protocol['checkpoint_sha256'],
            'manual_reviewed_old_model_images': {str((OUT/name).relative_to(ROOT)).replace('\\', '/'): sha(OUT/name) for name in viewed},
            'manual_findings': 'Old trajectory fits approximately; dynamic fragmentation/elongation and blurry static boundaries remain; no parameter updates or cleanup comparison.',
            'source_sha256': sha(Path(__file__))}
        write(REPORT/'final_review.json', result)
        write(RUNS/'final_status.json', result)
        write(RUNS/'state.json', {'status': result['status'], 'pending_review': False,
            'optimizer_updates': 0, 'final_review_sha256': sha(REPORT/'final_review.json')})
        pipeline = read(RUNS/'pipeline.json')
        pipeline.update(status=result['status'], stage='complete', pending_review=False,
            final_review_sha256=sha(REPORT/'final_review.json'), worker_pid_is_historical=True)
        write(RUNS/'pipeline.json', pipeline)
        print('J0_FINAL_REVIEW_COMPLETE_NO_UPDATE', flush=True)
    finally:
        for stream in reversed(locks):
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1); stream.close()


if __name__ == '__main__':
    main()
