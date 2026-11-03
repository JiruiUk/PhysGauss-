"""Read-only MonST3R source/camera audit; imports no downloaded model code."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from output.short_clip_diagnostic.camera_math import intrinsic, project, w2c


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tmp/monst3r_source_audit'
REPORT = ROOT / 'experiments/staged_reconstruction/monst3r_geometry'
COMMIT = '574cc77ad278bad582f470e5382624e01f8769a7'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method(tree, class_name, name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def main():
    target = REPORT / 'source_camera_audit.json'
    assert not target.exists(), 'Preserve earlier audit receipts'
    optimizer = ast.parse((SOURCE / 'optimizer.py').read_text(encoding='utf-8'))
    base = ast.parse((SOURCE / 'base_opt.py').read_text(encoding='utf-8'))
    # Inspect syntax only: do not execute foreign imports or functions.
    alignment = method(base, 'BasePCOptimizer', 'compute_global_alignment')
    known = next(n for n in ast.walk(alignment) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'preset_pose')
    assert next(k.value.value for k in known.keywords if k.arg == 'requires_grad') is True
    pose = method(optimizer, 'PointCloudOptimizer', 'preset_pose')
    assert ast.literal_eval(pose.args.defaults[-1]) is False
    intr = method(optimizer, 'PointCloudOptimizer', 'preset_intrinsics')
    assert any(isinstance(n, ast.Attribute) and n.attr == 'preset_focal' for n in ast.walk(intr))
    assert any(isinstance(n, ast.Attribute) and n.attr == 'optimize_pp' for n in ast.walk(intr))
    window = method(optimizer, 'PointCloudOptimizer', 'load_window_params')
    assert any(isinstance(n, ast.Attribute) and n.attr == 'requires_grad_' for n in ast.walk(window))

    metadata_path = ROOT / 'data/derived/kinematic_resplit/transforms_train.json'
    metadata = json.loads(metadata_path.read_text())
    original = (1600, 900)
    processed = (512, 288)
    ratio = processed[0] / original[0]
    # PIL resizing maps pixel centers, not pixel corners.
    image_transform = np.array([[ratio, 0, (ratio - 1) / 2],
                                [0, ratio, (ratio - 1) / 2], [0, 0, 1]])
    full_k = intrinsic(metadata['camera_angle_x'], 'metadata', *original)
    small_k = image_transform @ full_k
    np.testing.assert_allclose(small_k, intrinsic(metadata['camera_angle_x'], 'metadata', *processed), atol=1e-12)
    pixels = np.array([[0, 0], [1599, 899], [799.5, 449.5], [256, 128], [1200, 700.]])
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(full_k).T
    points = rays * np.array([1., 2., 3., 7., 12.])[:, None]
    rows = []
    for frame in metadata['frames']:
        view = w2c(frame['transform_matrix'], 'metadata_cv')
        c2w_cv = np.linalg.inv(view)
        gl = np.asarray(frame['transform_matrix'])
        np.testing.assert_allclose(c2w_cv, gl @ np.diag([1., -1., -1., 1.]), atol=1e-12)
        world = points @ c2w_cv[:3, :3].T + c2w_cv[:3, 3]
        uv, z = project(world, view, small_k)
        expected = (np.c_[pixels, np.ones(len(pixels))] @ image_transform.T)[:, :2]
        error = float(np.max(np.abs(uv - expected)))
        assert error < 1e-9 and (z > 0).all()
        recovered = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(small_k).T * z[:, None]
        np.testing.assert_allclose(recovered, points, atol=1e-12)
        rows.append({'frame': int(Path(frame['file_path']).stem), 'pixel_roundtrip_max_abs': error})

    dependencies = {}
    for package in ['torch', 'torchvision', 'roma', 'einops', 'huggingface_hub', 'timm', 'cv2', 'trimesh', 'evo', 'sam2']:
        dependencies[package] = importlib.util.find_spec(package) is not None
    REPORT.mkdir(parents=True, exist_ok=True)
    result = {
        'official_repository': 'https://github.com/Junyi42/monst3r',
        'official_commit': COMMIT,
        'downloaded_source_sha256': {p.name: digest(p) for p in sorted(SOURCE.iterdir()) if p.is_file()},
        'audit_script_sha256': digest(Path(__file__)),
        'metadata_sha256': digest(metadata_path),
        'known_poses_default_alignment_reenables_pose_gradients': True,
        'preset_pose_default_requires_grad': False,
        'preset_intrinsics_averages_fx_fy': True,
        'preset_intrinsics_sets_pp_only_when_optimize_pp_true': True,
        'window_mode_copies_can_reenable_pose_gradients': True,
        'metadata_GL_to_MonST3R_CV_c2w': 'metadata_c2w @ diag(1,-1,-1,1)',
        'native_size': original, 'processed_size': processed,
        'no_512_crop_needed_for_1600x900': True,
        'pixel_center_resize_transform': image_transform.tolist(),
        'processed_K': small_k.tolist(),
        'default_W_over_2_H_over_2_principal_point_offset_pixels': [.5, .5],
        'all_training_camera_roundtrips_passed': True, 'camera_rows': rows,
        'existing_environment_dependency_availability': dependencies,
        'downloaded_model_code_imported_or_executed': False,
        'RGB_read': False, 'optimizer_updates': 0,
        'weights_downloaded': False, 'geometry_qualified_for_training': False,
        'scope': 'Syntax and camera convention only; no assertion of model quality, runtime compatibility, frozen-camera optimization or novel-view completeness.'}
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('MONST3R_SOURCE_CAMERA_AUDIT', max(r['pixel_roundtrip_max_abs'] for r in rows), dependencies, flush=True)


if __name__ == '__main__':
    main()
