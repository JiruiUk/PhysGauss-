"""Preserve failed I9 admission and record actually inspected evidence."""
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.relative_strain.freeze import verify


def main():
    exp = ROOT / 'experiments/staged_reconstruction/relative_strain'
    target = exp / 'visual_review.json'; assert not target.exists()
    verify()
    admission_path = exp / 'warmup_admission.json'; before = sha(admission_path)
    admission = read(admission_path); assert not admission['passed']
    state = read(exp / 'warmup_state_audit.json'); assert state['passed']
    assert state['arms']['strain']['actual_optimizer_updates'] == 16000
    assert state['arms']['strain']['physics_calls'] == 0
    checkpoint = ROOT / 'runs/pidg_staged_reconstruction/relative_strain/formal/strain/checkpoint.pt'
    assert sha(checkpoint) == state['arms']['strain']['checkpoint_sha256']
    assert '18 passed' in (exp / 'final_unit_tests.log').read_text(encoding='utf-8', errors='replace')
    folder = ROOT / 'output/staged_reconstruction/relative_strain/warmup'
    inspected = {
        'fixed188_t0043_native.png': 'Large hollow/missing surfaces, flattened separate lobes and detached upper fragments.',
        'fixed188_t0146_native.png': 'Major hollow/missing surfaces and detached flattened support persist.',
        'fixed86_t0086_native.png': 'Upper hollow sheets/fragments and stretched or blurred support; object completeness fails.'}
    comparison = ROOT / 'output/staged_reconstruction/relative_strain/compare_warmup16000/comparison.png'
    write(target, {'checkpoint_sha256': sha(checkpoint), 'native_images_reviewed': True,
        'actually_inspected_native_images': { (folder / name).relative_to(ROOT).as_posix(): {'sha256': sha(folder / name), 'observation': text} for name, text in inspected.items()},
        'actually_inspected_comparison': {'path': comparison.relative_to(ROOT).as_posix(), 'sha256': sha(comparison)},
        'all_exported_native_images_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(folder.glob('*.png'))},
        'visual_completeness_passed': False, 'failed_numerical_admission_preserved': True,
        'admission_sha256': before, 'formal_physics_admitted': False,
        'six_camera_nine_time_screen_completed': False, 'full1200_acceptance_scan_completed': False,
        'test_RGB_read': False, 'original_thresholds_changed': False})
    assert sha(admission_path) == before
    write(exp / 'progress.json', {'mode': 'formal', 'arm': 'strain', 'actual_updates': 16000,
        'target': 22000, 'target_scope': 'Original16k warmup plus conditional6k physics',
        'acceptance': 'failed_original_warmup_static_preservation_and_visual_completeness',
        'physics_updates': 0, 'training_terminal': True,
        'goal_achieved': False})
    print('I9_FAILED_WARMUP_EVIDENCE_FINALIZED', sha(checkpoint), flush=True)


if __name__ == '__main__':
    main()
