"""Preregister a train-only wider-pair/known-camera scale qualification."""
import argparse
from pathlib import Path
from ..prepare_data import ROOT, DATA, read, write, sha

EXP = ROOT / 'experiments/staged_reconstruction/calibrated_pointmaps'
WORK = ROOT / 'runs/pidg_staged_reconstruction/calibrated_pointmaps'
PREVIOUS = ROOT / 'experiments/staged_reconstruction/monst3r_geometry'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--enable-calibrated-pointmap-pilot', action='store_true'); args = p.parse_args()
    if not args.enable_calibrated_pointmap_pilot:
        raise SystemExit('Calibrated pointmaps are an opt-in research pilot')
    assert not (EXP / 'protocol.json').exists()
    assert '10 passed' in (EXP / 'unit_tests.log').read_text(encoding='utf-8', errors='replace')
    metadata = read(DATA / 'transforms_train.json')
    ids = sorted(int(Path(f['file_path']).stem) for f in metadata['frames'])
    anchors = [2, 43, 86, 146, 188, 200]; partners = {}
    for anchor in anchors:
        offsets = [9, 18] if anchor < 20 else ([-9, -18] if anchor > 181 else [-9, 9])
        selected = []
        for offset in offsets:
            selected.append(min((n for n in ids if n != anchor and n not in selected), key=lambda n: (abs(n-anchor-offset), n)))
        partners[str(anchor)] = selected
    old = read(PREVIOUS / 'pilot_protocol.json')
    gates = {key: old[key] for key in ['held_static_median_max', 'held_static_p90_max', 'static_support_min',
        'dynamic_confidence_min', 'dynamic_confident_fraction_min', 'dynamic_extrapolation_fraction_max',
        'dynamic_partner_depth_median_max', 'dynamic_partner_depth_p90_max',
        'dynamic_self_projection_median_max_pixels', 'dynamic_self_projection_p90_max_pixels']}
    protocol = {'enabled_by_default': False, 'pilot': 'Known-camera dynamic pointmap scale, wider temporal baselines',
        'anchors': anchors, 'partners': partners, 'input_size': [512, 288],
        'pair_rule': 'Actual frame targets +/-9; first anchor +9,+18, last anchors -9,-18; nearest unique available training frame, lower-ID tie break',
        'data_metadata_sha256': sha(DATA / 'transforms_train.json'), 'source_RGB': 'training120 only, no missing RGB interpolation',
        'weights_manifest_sha256': sha(PREVIOUS / 'asset_weights.json'),
        'vendor_manifest_sha256': sha(PREVIOUS / 'pair_vendor.json'),
        'old_protocol_sha256': sha(PREVIOUS / 'pilot_protocol.json'), 'old_pilot_gates_unchanged': gates,
        'known_camera_conversion_audit_sha256': sha(PREVIOUS / 'source_camera_audit.json'),
        'parent_sha256': '8c767b7835eee936e49989369d7bc0277d7567d3aedaa7385c06654284b68ad0',
        'arms': ['wide_pairs_static_scale_control', 'wide_pairs_known_camera_scale'],
        'control_scale': 'One scalar per pair from every third confident first-image static pixel; held-out others',
        'camera_scale': 'b_j cross (R_ji (s Q_j) + t_ji) = 0; fit second-image observed rays without static correspondence assumptions',
        'fit_pixels': '(row+2*column)%3 == 0 on eroded observed alpha>0.95 and confidence>=3; held-out complement',
        'scale_fit_min': 128, 'scale_held_min': 128, 'robust_cauchy_relative_baseline': .05, 'IRLS_iterations': 8,
        'median_predicted_ray_parallax_min_degrees': .5,
        'held_second_projection_median_max_pixels': 2., 'held_second_projection_p90_max_pixels': 5.,
        'camera_or_model_optimization': False, 'static_reference_is_model_conditioned_not_GT': True,
        'no_pointmap_reprojection_or_focal_warp_to_force_a_gate': True,
        'known_cameras_are_constraints_not_invented_synchronized_GT': True,
        'PIDG_training_targets_or_updates': 0, 'passing_is_not_dynamic_geometry_truth_or_completeness': True}
    write(EXP / 'protocol.json', protocol)
    files = [*sorted(Path(__file__).parent.glob('*.py')), ROOT / 'tests/test_calibrated_pointmaps.py',
             EXP / 'research.md', EXP / 'protocol.json']
    write(EXP / 'source_freeze.json', {p.relative_to(ROOT).as_posix(): sha(p) for p in files})
    print('CALIBRATED_POINTMAP_PROTOCOL_FIXED', partners, gates, flush=True)


if __name__ == '__main__':
    main()
