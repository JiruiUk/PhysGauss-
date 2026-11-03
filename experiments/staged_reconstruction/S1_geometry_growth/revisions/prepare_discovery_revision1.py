"""Calibrated dense-depth discovery, explicit planar completion and surface fusion."""
import argparse
from pathlib import Path
import msvcrt
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.segmentation import felzenszwalb
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..automatic_static.discover import rgb
from output.short_clip_diagnostic.camera_math import w2c, intrinsic, project
from .geometry import (camera_center, select_views, patch_features, sweep, depth_consistency,
                       fuse_voxels, fit_plane, plane_intersections)

REPORT = ROOT/'experiments/staged_reconstruction/S1_geometry_growth'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_geometry_growth'
OUT = ROOT/'output/staged_reconstruction/S1_geometry_growth'


def verify():
    for path, expected in read(REPORT/'source_freeze.json').items(): assert sha(ROOT/path) == expected, path


def world_points(uv, depth, view, k):
    local = np.c_[uv, np.ones(len(uv))]@np.linalg.inv(k).T*depth[:, None]
    inverse = np.linalg.inv(view)
    return local@inverse[:3, :3].T+inverse[:3, 3]


def inputs(protocol):
    meta = read(DATA/'transforms_train.json')
    frames = sorted(meta['frames'], key=lambda f: int(Path(f['file_path']).stem))
    ids = [int(Path(f['file_path']).stem) for f in frames]
    images = np.stack([rgb(f, protocol['dense_width']) for f in frames])
    views = np.asarray([w2c(f['transform_matrix'], 'metadata_cv') for f in frames])
    k = intrinsic(meta['camera_angle_x'], 'metadata', protocol['dense_width'], protocol['dense_width']*9//16)
    return meta, frames, ids, images, views, k


def scan(protocol, frames, ids, images, views, k):
    directory = RUNS/'raw_depths'; directory.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol': sha(REPORT/'protocol.json'),
        'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['prepare.py', 'geometry.py']},
        'metadata_sha256': sha(DATA/'transforms_train.json'),
        'images': {f['file_path']: sha(DATA/(f['file_path']+'.png')) for f in frames}}
    if (REPORT/'scan_inputs.json').exists(): assert manifest == read(REPORT/'scan_inputs.json')
    else: write(REPORT/'scan_inputs.json', manifest)
    device = 'cuda'
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
    features, texture = patch_features(tensor, protocol['patch_kernel']); del tensor
    view_tensor = torch.tensor(views, dtype=torch.float32, device=device)
    k_tensor = torch.tensor(k, dtype=torch.float32, device=device)
    center = camera_center(views); origins = np.linalg.inv(views)[:, :3, 3]
    radius = float(np.median(np.linalg.norm(origins-center, axis=1)))
    near, far = radius*protocol['depth_near_camera_distance_factor'], radius*protocol['depth_far_camera_distance_factor']
    depths = 1/torch.linspace(1/near, 1/far, protocol['depth_samples'], device=device)
    rows = []
    for i, number in enumerate(ids):
        path = directory/f'{number:04d}.npz'
        if path.exists():
            with np.load(path) as data: rows.append({'frame': number, 'photo_candidates': int((data['depth'] > 0).sum()), 'sources': data['sources'].tolist()})
            continue
        sources = select_views(i, views, ids, min_angle=protocol['minimum_camera_angle'], max_angle=protocol['maximum_camera_angle'])
        h, width = images.shape[1:3]; depth_map = np.zeros((h, width), np.float32)
        cost_map = np.ones_like(depth_map); margin_map = np.zeros_like(depth_map)
        if sources:
            coordinates = torch.nonzero(texture[i], as_tuple=False)[:, [1, 0]]
            inside = (coordinates[:, 0] >= 2)&(coordinates[:, 0] < width-2)&(coordinates[:, 1] >= 2)&(coordinates[:, 1] < h-2)
            coordinates = coordinates[inside]
            for start in range(0, len(coordinates), protocol['pixel_chunk']):
                uv = coordinates[start:start+protocol['pixel_chunk']].float()
                _, z, good, cost, margin, _ = sweep(uv, features, texture, view_tensor, k_tensor, depths, i, sources,
                    protocol['uniqueness_margin'], protocol['photometric_cost_limit'])
                xy = uv[good].long().cpu().numpy()
                depth_map[xy[:, 1], xy[:, 0]] = z[good].cpu().numpy()
                cost_map[xy[:, 1], xy[:, 0]] = cost[good].cpu().numpy()
                margin_map[xy[:, 1], xy[:, 0]] = margin[good].cpu().numpy()
        np.savez_compressed(path, depth=depth_map, cost=cost_map, margin=margin_map, sources=np.asarray(sources, int))
        rows.append({'frame': number, 'photo_candidates': int((depth_map > 0).sum()), 'sources': sources})
        write(RUNS/'state.json', {'status': 'dense_discovery', 'frame': number, 'completed_frames': i+1, 'pid': os.getpid()})
        if i % 5 == 0: print('DENSE_STATIC_SCAN', i+1, '/', len(ids), 'frame', number, 'photo', int((depth_map > 0).sum()), flush=True)
        if (RUNS/'stop.request').exists(): return False
    write(REPORT/'scan.json', {'frames': rows, 'camera_aim_center': center.tolist(), 'camera_median_distance': radius,
        'depth_range': [near, far], 'source_input_manifest_sha256': sha(REPORT/'scan_inputs.json'),
        'raw_depth_sha256': {p.name: sha(p) for p in directory.glob('*.npz')}})
    return True


def checked_maps(depths, sources, views, k, ids, protocol, origins=None):
    checked, support_maps, reports = [], [], []
    for i, (depth, source_list) in enumerate(zip(depths, sources)):
        yy, xx = np.nonzero(depth > 0); uv = np.c_[xx, yy]
        xyz = world_points(uv, depth[yy, xx], views[i], k)
        support = np.zeros((len(xyz), len(ids)), bool); support[:, i] = True
        errors = []
        for j in source_list:
            good, error = depth_consistency(xyz, uv, depths[j], views[j], views[i], k,
                protocol['depth_relative_tolerance'], protocol['reprojection_limit_px'])
            support[:, j] = good; errors += error[good].tolist()
        span = np.zeros(len(xyz), int)
        for n, row in enumerate(support):
            observed = np.asarray(ids)[row]; span[n] = observed.max()-observed.min()
        keep = (support.sum(1) >= 1+protocol['depth_consistent_sources'])&(span >= protocol['minimum_source_span'])
        output = np.zeros_like(depth); output[yy[keep], xx[keep]] = depth[yy[keep], xx[keep]]
        checked.append(output); support_maps.append((uv[keep], support[keep]))
        reports.append({'frame': ids[i], 'photo_pixels': len(xyz), 'depth_checked_pixels': int(keep.sum()),
                        'accepted_link_reprojection_median': float(np.median(errors)) if errors else None})
    return checked, support_maps, reports


def planar_completion(images, raw_checked, views, k, ids, sources, protocol):
    filled = [d.copy() for d in raw_checked]; plane_flags = [np.zeros_like(d, bool) for d in raw_checked]
    records = []
    blurred = np.stack([cv2.medianBlur((im*255).round().astype('uint8'), 3).astype(np.float32)/255 for im in images])
    height, width = images.shape[1:3]
    for i, depth in enumerate(raw_checked):
        segments = felzenszwalb(images[i], scale=100, sigma=.8, min_size=60)
        for label in np.unique(segments):
            yy, xx = np.nonzero((segments == label)&(depth > 0))
            if len(xx) < protocol['plane_min_points']: continue
            xyz = world_points(np.c_[xx, yy], depth[yy, xx], views[i], k)
            plane = fit_plane(xyz, protocol['plane_min_points'], protocol['plane_rms_limit'],
                              protocol['plane_second_eigenvalue_min'], protocol['plane_smallest_to_second_ratio_max'])
            if plane is None: continue
            y, x = np.nonzero((segments == label)&(depth == 0)&(images[i].mean(2) > .025))
            if not len(x): continue
            points, z, stable = plane_intersections(np.c_[x, y], plane, views[i], k)
            stable &= (z > .2)&np.isfinite(z)
            votes = np.zeros((len(points), len(sources[i])), bool)
            for col, j in enumerate(sources[i]):
                uv, source_z = project(points, views[j], k)
                good = stable&(source_z > .2)&np.isfinite(uv).all(1)&(uv[:, 0] >= 1)&(uv[:, 0] < width-1)&(uv[:, 1] >= 1)&(uv[:, 1] < height-1)
                rows = np.flatnonzero(good); xy = np.rint(uv[rows]).astype(int)
                color = blurred[j, xy[:, 1], xy[:, 0]]
                compatible = (np.mean(np.abs(color-blurred[i, y[rows], x[rows]]), 1) <= protocol['plane_color_limit'])&(color.mean(1) > .025)
                known_z = raw_checked[j][xy[:, 1], xy[:, 0]]
                # Occlusion/depth disagreement supplies no positive vote.
                compatible &= (known_z == 0)|(np.abs(known_z-source_z[rows]) <= protocol['depth_relative_tolerance']*source_z[rows])
                votes[rows, col] = compatible
            accepted = stable&(votes.sum(1) >= protocol['plane_min_color_views'])&(votes[:, 4:].sum(1) >= 1)
            filled[i][y[accepted], x[accepted]] = z[accepted]
            plane_flags[i][y[accepted], x[accepted]] = True
            if accepted.any(): records.append({'frame': ids[i], 'region': int(label), 'raw_anchor_pixels': len(xx),
                'filled_pixels_before_final_depth_check': int(accepted.sum()), 'held_plane_rms': plane['held_rms'],
                'normal': plane['normal'].tolist(), 'center': plane['center'].tolist(), 'assumption': 'local_planarity_prior_not_independent_depth_GT'})
        if i % 20 == 0: print('STATIC_PLANAR_COMPLETION', i, 'planes', len(records), flush=True)
    return filled, plane_flags, records


def fuse(protocol, meta, frames, ids, images, views, k):
    assert not (REPORT/'discovery.json').exists(), 'Preserve completed discovery'
    manifest = read(REPORT/'scan_inputs.json'); assert manifest['protocol'] == sha(REPORT/'protocol.json')
    for path, expected in manifest['sources'].items(): assert sha(ROOT/path) == expected, path
    scan_receipt = read(REPORT/'scan.json')
    depths, sources = [], []
    for number in ids:
        path = RUNS/'raw_depths'/f'{number:04d}.npz'; assert sha(path) == scan_receipt['raw_depth_sha256'][path.name]
        with np.load(path) as data: depths.append(data['depth'].copy()); sources.append(data['sources'].tolist())
    first_checked, _, first_report = checked_maps(depths, sources, views, k, ids, protocol)
    if protocol['enable_planar_completion']:
        completed, flags, plane_records = planar_completion(images, first_checked, views, k, ids, sources, protocol)
    else: completed, flags, plane_records = first_checked, [np.zeros_like(d, bool) for d in first_checked], []
    final, supports, final_report = checked_maps(completed, sources, views, k, ids, protocol)
    positions, colors, evidence, plane_origin = [], [], [], []
    masks, geometries, panels = {}, {}, []
    for i, number in enumerate(ids):
        uv, support = supports[i]; xy = uv.astype(int)
        p = world_points(uv, final[i][xy[:, 1], xy[:, 0]], views[i], k)
        positions.append(p); colors.append(images[i][xy[:, 1], xy[:, 0]]); evidence.append(support)
        plane_origin.append(flags[i][xy[:, 1], xy[:, 0]])
        cap = np.where(final[i] > 0, np.where(flags[i], protocol['plane_fill_geometry_cap'], .95), 0.).astype(np.float32)
        masks[str(number)] = cap.copy(); geometries[str(number)] = cap
        if number in [2, 27, 43, 86, 166, 188, 200]:
            panel = Image.new('RGB', (1200, 250), '#161922')
            for col, (mask, title) in enumerate([(None, 'Observed RGB'), (first_checked[i] > 0, 'Independent dense depth'), (final[i] > 0, 'After verified planar completion')]):
                im = (images[i]*255).round().astype('uint8')
                if mask is not None: im[mask] = .35*im[mask]+.65*np.array([0, 255, 100])
                panel.paste(Image.fromarray(im), (col*400, 25)); ImageDraw.Draw(panel).text((col*400+5, 5), f't{number} {title}', fill='white')
            panels.append(np.asarray(panel))
    xyz, col, sup = np.concatenate(positions), np.concatenate(colors), np.concatenate(evidence)
    assert len(xyz) >= 100, 'Insufficient dense evidence; preserve unknown, no fallback to static colors'
    reservoir_xyz, reservoir_rgb, reservoir_support = fuse_voxels(xyz, col, sup, protocol['reservoir_voxel_spacing'])
    _, inverse = np.unique(np.floor(xyz/protocol['reservoir_voxel_spacing']).astype(np.int64), axis=0, return_inverse=True)
    raw_presence = np.bincount(inverse, weights=(~np.concatenate(plane_origin)).astype(float), minlength=len(reservoir_xyz)) > 0
    quality_order = np.lexsort((np.arange(len(reservoir_xyz)), -reservoir_support.sum(1), ~raw_presence))
    coarse = np.floor(reservoir_xyz[quality_order]/protocol['initial_voxel_spacing']).astype(np.int64)
    _, first = np.unique(coarse, axis=0, return_index=True); initial_ids = np.sort(quality_order[first])
    if len(initial_ids) > protocol['initial_max_points']:
        # Uniform spatial representatives; never favor a late high-count cluster.
        initial_ids = np.sort(np.random.default_rng(protocol['seed']).choice(initial_ids, protocol['initial_max_points'], False))
    assert len(initial_ids) >= 100
    np.savez_compressed(RUNS/'reservoir.npz', xyz=reservoir_xyz, rgb=reservoir_rgb, support=reservoir_support,
                        plane_only=~raw_presence, frame_ids=np.asarray(ids))
    np.savez_compressed(RUNS/'initialization.npz', xyz=reservoir_xyz[initial_ids], rgb=reservoir_rgb[initial_ids],
                        spacing=np.asarray(protocol['model_spacing']), point_ids=initial_ids)
    np.savez_compressed(RUNS/'initial_confidence.npz', **masks)
    np.savez_compressed(RUNS/'geometric_confidence.npz', **geometries)
    np.savez_compressed(RUNS/'verified_depths.npz', **{str(n): d for n, d in zip(ids, final)})
    OUT.mkdir(parents=True, exist_ok=True); Image.fromarray(np.concatenate(panels)).save(OUT/'geometry_proposals.png')
    from ..cross_temporal_static.core import summary
    write(REPORT/'discovery.json', {'raw_depth_checks': first_report, 'completed_depth_checks': final_report,
        'verified_plane_priors': plane_records, 'plane_prior_pixels_after_final_check': int(sum((flag&(depth > 0)).sum() for flag, depth in zip(flags, final))),
        'dense_surface_voxels': len(reservoir_xyz), 'raw_evidence_surface_voxels': int(raw_presence.sum()),
        'plane_only_surface_voxels': int((~raw_presence).sum()), 'initial_points': len(initial_ids),
        'confidence': summary(masks), 'time_thirds': {str(t): {'verified_pixels': sum(int((d > 0).sum()) for n, d in zip(ids, final) if min((n-1)*3//200, 2) == t),
            'frames_with_support': sum(bool((d > 0).any()) for n, d in zip(ids, final) if min((n-1)*3//200, 2) == t)} for t in range(3)},
        'inputs_known_static_colors': False, 'independent_dataset_geometry_GT': False,
        'image_sha256': manifest['images'], 'artifacts': {p.name: sha(p) for p in [RUNS/n for n in ['reservoir.npz', 'initialization.npz', 'initial_confidence.npz', 'geometric_confidence.npz', 'verified_depths.npz']]}})
    write(RUNS/'state.json', {'status': 'geometry_prepared_pending_review', 'initial_points': len(initial_ids), 'reservoir_points': len(reservoir_xyz)})
    print('DENSE_STATIC_PREPARED', len(initial_ids), 'initial', len(reservoir_xyz), 'reservoir', summary(masks), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-adaptive-static', action='store_true')
    parser.add_argument('--phase', choices=['scan', 'fuse', 'all'], default='all'); args = parser.parse_args()
    if not args.enable_adaptive_static: raise SystemExit('Adaptive static reconstruction is opt-in')
    assert not (RUNS/'stop.request').exists(), 'Saved stop request; do not resume'
    RUNS.mkdir(parents=True, exist_ok=True); cv2.setNumThreads(2); torch.set_num_threads(4)
    lock = (RUNS/'experiment.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        protocol = read(REPORT/'protocol.json'); data = inputs(protocol)
        if args.phase in ['scan', 'all'] and not scan(protocol, data[1], data[2], data[3], data[4], data[5]): return
        if args.phase in ['fuse', 'all']: fuse(protocol, *data)
    finally: lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__': main()
