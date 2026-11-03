"""Freeze a separate union experiment with no instance initialization inputs."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from ..prepare_data import ROOT, REPORT, RUNS, DATA, read, write, sha
from output.short_clip_diagnostic.camera_math import w2c, intrinsic
from .core import initialize_union

UNION_REPORT = REPORT/'S3_union'
UNION_RUNS = RUNS/'S3_union'


def verify_sources():
    for name, expected in read(UNION_REPORT/'source_freeze.json').items():
        assert sha(ROOT/name) == expected, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-union-stage', action='store_true')
    args = parser.parse_args()
    if not args.enable_union_stage:
        raise SystemExit('S3-U is opt-in; pass --enable-union-stage')
    path = UNION_REPORT/'protocol.json'
    if path.exists():
        verify_sources()
        print('UNION_ALREADY_PREPARED', flush=True)
        return
    accepted = read(REPORT/'S1_final_verification.json')
    assert accepted['numeric_admission'] and sha(RUNS/'S1_static/checkpoint.pt') == accepted['checkpoint_sha256']
    meta_file = DATA/'transforms_train.json'
    meta = read(meta_file)
    assert len(meta['frames']) == 120
    first = meta['frames'][0]
    width = 800
    image_path = DATA/(first['file_path']+'.png')
    rgba = np.asarray(Image.open(image_path).convert('RGBA').resize((width, width*9//16), Image.Resampling.BILINEAR), dtype=np.float32)/255
    static_path = RUNS/'static_metadata_initialization.npz'
    static = np.load(static_path)['xyz']
    # Fixed total point budget matches the previous aggregate, with no per-instance allotment.
    initial = initialize_union(rgba, w2c(first['transform_matrix'], 'metadata_cv'),
        intrinsic(meta['camera_angle_x'], 'metadata', width, width*9//16), static, 15507, 20261104)
    UNION_RUNS.mkdir(parents=True, exist_ok=True)
    initial_path = UNION_RUNS/'dynamic_union.npz'
    assert not initial_path.exists(), 'Preserve existing initialization'
    np.savez_compressed(initial_path, **initial)
    receipt = {'initialization_file': initial_path.relative_to(ROOT).as_posix(), 'initialization_sha256': sha(initial_path),
        'training_metadata_sha256': sha(meta_file), 'training_image_sha256': {f['file_path']: sha(DATA/(f['file_path']+'.png')) for f in meta['frames']},
        'static_depth_prior_sha256': sha(static_path), 'initialization_frame': first['file_path'],
        'dynamic_points': len(initial['xyz']), 'input_instance_labels': False, 'input_physical_body_number': None,
        'number_inference': False, 'depth_prior': 'Per-pixel nearby static front depth minus0.25 with uniform[-0.12,0.12]; not depth GT',
        'appearance_prior': 'Same yellow foreground threshold as previous image objectives; one union without connected components',
        'test_RGB_or_geometry_used': False}
    write(UNION_REPORT/'data_receipt.json', receipt)
    protocol = {'stage': 'S3-U', 'enabled_by_default': False, 'seed': 20261104,
        'representation': 'One unified canonical dynamic population; no physical body numbering or inference',
        'static_checkpoint_sha256': accepted['checkpoint_sha256'], 'static_frozen': True,
        'data_receipt_sha256': sha(UNION_REPORT/'data_receipt.json'), 'camera_contract': 'metadata_cv',
        'dynamic_point_budget': 15507,
        'grid_args': {'canonical_num_levels': 12, 'canonical_log2_hashmap_size': 17, 'canonical_desired_resolution': 256,
                      'deform_num_levels': 16, 'deform_log2_hashmap_size': 17, 'deform_desired_resolution': [32, 32, 16]},
        'net_args': {'depth': 1, 'width': 128},
        'capacity_policy': 'One grid with aggregate hash-table cap comparable to prior split grids; one shared MLP/velocity/stress field. Actual parameter counts reported.',
        'warmup_updates': 16000, 'warmup_width_schedule': [[1, 800], [12001, 1600]],
        'physics_updates': 6000, 'physics_width': 1600,
        'image_objective': '0.8 L1 +0.2 SSIM +0.5 foreground RGB +0.5 static RGB +2.0 one-sided union coverage',
        'kinematic_weight': .01, 'CMR_weight': .00001, 'physics_every': 10, 'shared_gradient_cap': .1,
        'physics_samples_global': 64, 'time_step': 1/199, 'rate_coefficient_fixed': 1., 'density_fixed': 1.,
        'neighborhood_policy': 'Current-world kNN inside the dynamic population only; no physical-boundary exclusion is available',
        'time_units': 'sequence normalized; acquisition seconds unknown',
        'checkpoint_interval': 500, 'validation_interval': 2000,
        'warmup_admission': {'dynamic_PSNR_min': 20., 'dynamic_only_alpha_min': .85, 'static_PSNR_drop_max': 1.},
        'sparse_tracks': False, 'contact': False, 'constitutive': False, 'unknown_force': False, 'densification': False,
        'final_preselected': True, 'comparison_limit': 'Fresh union initialization and shared architecture; not a single-switch ablation from an instance-conditioned checkpoint',
        'success_limit': 'No number estimation, physical identities, independent geometry/stress GT or calibrated contact claim'}
    write(path, protocol)
    files = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in Path(__file__).parent.glob('*.py')}
    files['tests/test_staged_union.py'] = sha(ROOT/'tests/test_staged_union.py')
    for row in read(REPORT/'completion_integrity.json')['source_receipts']:
        value = read(ROOT/row['receipt'])
        for name, expected in value.get('sha256', value).items():
            assert sha(ROOT/name) == expected, name
            files[name] = expected
    for p in [path, UNION_REPORT/'data_receipt.json', initial_path, meta_file, static_path]:
        files[p.relative_to(ROOT).as_posix()] = sha(p)
    for frame in meta['frames']:
        p = DATA/(frame['file_path']+'.png')
        files[p.relative_to(ROOT).as_posix()] = sha(p)
    write(UNION_REPORT/'source_freeze.json', files)
    print('UNION_PREPARED', len(initial['xyz']), 'points; no instance inputs', flush=True)


if __name__ == '__main__':
    main()
