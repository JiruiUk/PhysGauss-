"""Read-only S1 preparation: fixed-reference geometry and opacity drift in R0."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.kinematic_tuning.model import center_trajectory
from output.short_clip_diagnostic.camera_math import w2c, intrinsic, project

REPORT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def describe(values):
    values = np.asarray(values, dtype=float)
    assert values.size and np.isfinite(values).all()
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95)), 'max': float(values.max())}


def main():
    torch.set_num_threads(4)
    bind = read(ROOT/'experiments/static_pool/binding.json')
    old = read(ROOT/'experiments/scratch_integrity/data_binding.json')
    with np.load(bind['path'], allow_pickle=False) as z:
        ids = z['static_ids'].copy(); refs = z['reference_frames'].copy()
        warp = torch.tensor(z['d_xyz'].copy())
    with np.load(old['path'], allow_pickle=False) as z:
        anchor_ids = z['anchor_ids'].copy(); anchor_xyz = z['anchor_xyz'].copy()
    selected_index = {int(row): j for j, row in enumerate(ids)}
    assert all(int(row) in selected_index for row in anchor_ids)
    audit = read(ROOT/'experiments/transfer_integrity/static_image_audit.json')
    track_xyz = np.asarray([t['xyz'] for t in audit['tracks']])
    tracks = []
    for row, xyz in zip(anchor_ids, anchor_xyz):
        distances = np.linalg.norm(track_xyz-xyz, axis=1)
        j = int(distances.argmin()); assert distances[j] < 1e-5
        tracks.append((selected_index[int(row)], audit['tracks'][j]))
    meta = read(ROOT/'data/derived/kinematic_resplit/training/transforms_train.json')
    frames = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    assert set(refs).issubset(frames)
    k = intrinsic(meta['camera_angle_x'], 'legacy')
    checkpoints = {
        'initial15k_fixed_reference': ROOT/'runs/pidg_scratch_integrity/common_candidates/15000/rubber_duck/checkpoints/resume.pt',
        'H1_final50k_fixed_reference': ROOT/'runs/pidg_static_pool/pilot/H1_static_pool/rubber_duck/checkpoints/resume.pt',
    }
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in checkpoints.values()}
    inputs.update({str(Path(bind['path']).relative_to(ROOT)): sha(Path(bind['path'])),
                   str(Path(old['path']).relative_to(ROOT)): sha(Path(old['path']))})
    results = {}; arrays = {}
    for name, path in checkpoints.items():
        state = torch.load(path, map_location='cpu', weights_only=False)
        parameters = state['gaussian_parameters']
        xyz = parameters['_xyz'][ids]
        centers = center_trajectory(xyz, warp).numpy()
        opacity = parameters['_opacity'][ids].sigmoid().flatten().numpy()
        scale = parameters['_scaling'][ids].exp().numpy()
        arrays[name] = (centers, opacity, scale)
        errors = []; held = []
        for row, track in tracks:
            for index, (frame, uv) in enumerate(zip(track['frames'], track['uv'])):
                pred, depth = project(centers[row:row+1], w2c(frames[frame]['transform_matrix'], 'legacy'), k)
                error = float(np.linalg.norm(pred[0]-uv)) if depth[0] > .2 else 400.
                errors.append(error)
                if index % 2: held.append(error)
        results[name] = {'opacity': describe(opacity),
                         'opacity_below_001_fraction': float(np.mean(opacity < .01)),
                         'track_reprojection_px_400': describe(errors),
                         'triangulation_held_reprojection_px_400': describe(held),
                         'track_observations': len(errors)}
        del state, parameters
    initial, final = [arrays[name] for name in checkpoints]
    displacement = np.linalg.norm(final[0]-initial[0], axis=1)
    results['fixed_world_center_drift'] = describe(displacement)
    groups = []
    for frame in sorted(set(refs)):
        mask = refs == frame
        groups.append({'reference_frame': int(frame), 'rows': int(mask.sum()),
                       'initial_opacity_mean': float(initial[1][mask].mean()),
                       'final_opacity_mean': float(final[1][mask].mean()),
                       'final_opacity_below_001_fraction': float(np.mean(final[1][mask] < .01)),
                       'center_drift': describe(displacement[mask])})
    # Same fixed transform, independently evaluated on CPU, must reproduce R0.
    previous = read(ROOT/'experiments/static_pool/static_diagnostics.json')['H1_static_pool']
    assert abs(results['H1_final50k_fixed_reference']['track_reprojection_px_400']['mean'] -
               previous['static_image_track_reprojection_mean_px_400']) < 1e-4
    assert all(sha(ROOT/p) == expected for p, expected in inputs.items())
    with (REPORT/'selected_rows.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row','reference_frame','initial_opacity','final_opacity','world_center_drift',
                         'initial_x','initial_y','initial_z','final_x','final_y','final_z'])
        for j, row in enumerate(ids):
            writer.writerow([row,refs[j],initial[1][j],final[1][j],displacement[j],*initial[0][j],*final[0][j]])
    result = {'status': 'completed_read_only_S1_preparation', 'training_started': False,
              'S1_accepted': False, 'static_rows': len(ids), 'reference_frames': len(groups),
              'results': results, 'by_reference_frame': groups, 'inputs_sha256_unchanged': inputs,
              'audit_script_sha256': sha(Path(__file__)),
              'limits': ['37 conditional static tracks; not independent3D GT.',
                         'No static-only optimization or complete static/dynamic masks were supplied by R0.',
                         'This audit cannot certify the geometry of unobserved rows.']}
    (REPORT/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'static_rows':len(ids),'reference_frames':len(groups),'results':results},indent=2))


if __name__ == '__main__':
    main()
