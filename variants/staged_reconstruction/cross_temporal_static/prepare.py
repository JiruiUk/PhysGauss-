"""Training-only wide-baseline anchors and cross-temporal confidence proposals."""
import argparse
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from skimage.segmentation import felzenszwalb
from PIL import Image, ImageDraw
from ..prepare_data import ROOT, DATA, read, write, sha
from ..automatic_static.discover import rgb
from ..automatic_static.geometry import evidence
from output.short_clip_diagnostic.camera_math import w2c, intrinsic, project
from .core import mutual_matches, epipolar_errors, join_tracks, summary

REPORT = ROOT/'experiments/staged_reconstruction/S1_cross_temporal'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_cross_temporal'
OUT = ROOT/'output/staged_reconstruction/S1_cross_temporal'


def verify():
    for path, expected in read(REPORT/'source_freeze.json').items():
        assert sha(ROOT/path) == expected, path


def pixel_colors(image, xy):
    # Median3x3 reduces interpolation/outlier noise, without any color class.
    blurred = cv2.medianBlur((image*255).round().astype('uint8'), 3).astype(float)/255
    xy = np.rint(xy).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, image.shape[1]-1)
    xy[:, 1] = np.clip(xy[:, 1], 0, image.shape[0]-1)
    return blurred[xy[:, 1], xy[:, 0]]


def region_confidence(frame, image, anchors, incompatible, meta, protocol):
    """Only independent fixed-point evidence can seed a generic image region."""
    number = int(Path(frame['file_path']).stem)
    labels = felzenszwalb(image, scale=100, sigma=.8, min_size=60)
    n = labels.max()+1; height, width = labels.shape
    xyz = np.asarray([a['xyz'] for a in anchors])
    uv, depth = project(xyz, w2c(frame['transform_matrix'], 'metadata_cv'),
                        intrinsic(meta['camera_angle_x'], 'metadata', width, height))
    good = (depth > .2)&np.isfinite(uv).all(1)&(uv[:, 0] >= 1)&(uv[:, 0] < width-1)&(uv[:, 1] >= 1)&(uv[:, 1] < height-1)
    rows = np.flatnonzero(good); xy = np.rint(uv[rows]).astype(int)
    colors = pixel_colors(image, uv[rows])
    reference = np.asarray([anchors[i]['rgb'] for i in rows])
    consistent = np.mean(np.abs(colors-reference), 1) <= protocol['projection_color_limit']
    counts = np.bincount(labels[xy[consistent, 1], xy[consistent, 0]], minlength=n)
    observed_uv = []
    for anchor in anchors:
        if number in anchor['frames']:
            observed_uv.append(anchor['uv'][anchor['frames'].index(number)])
    direct = np.zeros(n, int)
    if observed_uv:
        q = np.rint(observed_uv).astype(int)
        valid = (q[:, 0] >= 0)&(q[:, 0] < width)&(q[:, 1] >= 0)&(q[:, 1] < height)
        direct = np.bincount(labels[q[valid, 1], q[valid, 0]], minlength=n)
    conflict = np.zeros(n, int)
    bad_uv = [r['uv'][r['frames'].index(number)] for r in incompatible if number in r['frames']]
    if bad_uv:
        q = np.rint(bad_uv).astype(int)
        valid = (q[:, 0] >= 0)&(q[:, 0] < width)&(q[:, 1] >= 0)&(q[:, 1] < height)
        conflict = np.bincount(labels[q[valid, 1], q[valid, 0]], minlength=n)
    initial = np.where((direct >= protocol['minimum_region_anchors'])&(conflict == 0), .95, 0.).astype(np.float32)
    geometry = np.maximum(initial, np.where((counts >= protocol['minimum_region_anchors'])&(conflict == 0), .85, 0.)).astype(np.float32)
    # Erode each admitted region boundary, never bridge mixed silhouettes.
    initial_map, geometry_map = initial[labels], geometry[labels]
    initial_map *= binary_erosion(initial_map > 0, iterations=1)
    geometry_map *= binary_erosion(geometry_map > 0, iterations=1)
    return initial_map, geometry_map


def initialize(anchors, geometry, frames, images, meta, protocol):
    xyz = np.asarray([a['xyz'] for a in anchors])
    radius = protocol['anchor_radius']; spacing = protocol['voxel_spacing']
    low, high = xyz.min(0)-radius, xyz.max(0)+radius
    axes = [np.arange(a, b+spacing/2, spacing, dtype=np.float32) for a, b in zip(low, high)]
    shape = tuple(map(len, axes)); assert np.prod(shape) < 3000000, shape
    points = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    near = cKDTree(xyz).query(points)[0] <= radius
    seen = np.zeros(len(points), int); positive = seen.copy()
    thirds = np.zeros((len(points), 3), int); colors = np.zeros((len(points), 3))
    k = intrinsic(meta['camera_angle_x'], 'metadata', 400, 225)
    for frame, image in zip(frames, images):
        number = int(Path(frame['file_path']).stem)
        uv, depth = project(points, w2c(frame['transform_matrix'], 'metadata_cv'), k)
        good = (depth > .2)&np.isfinite(uv).all(1)&(uv[:, 0] >= 0)&(uv[:, 0] < 400)&(uv[:, 1] >= 0)&(uv[:, 1] < 225)
        rows = np.flatnonzero(good); xy = uv[rows].astype(int)
        support = geometry[str(number)][xy[:, 1], xy[:, 0]] >= .5
        seen[rows] += 1; positive[rows] += support
        thirds[rows, min((number-1)*3//200, 2)] += support
        colors[rows] += image[xy[:, 1], xy[:, 0]]*support[:, None]
    keep = near&(positive >= 8)&(seen >= 12)&((thirds >= 2).sum(1) >= 2)
    volume = keep.reshape(shape)
    rows = np.flatnonzero((volume&~binary_erosion(volume)).reshape(-1))
    assert len(rows) >= 100, 'Insufficient geometry; preserve unknown, no label fallback'
    if len(rows) > protocol['max_static_points']:
        rows = np.sort(np.random.default_rng(protocol['seed']).choice(rows, protocol['max_static_points'], False))
    np.savez_compressed(RUNS/'initialization.npz', xyz=points[rows],
                        rgb=(colors[rows]/np.maximum(positive[rows, None], 1)).astype(np.float32), spacing=np.asarray(spacing))
    return {'static_points': len(rows), 'bounds': [low.tolist(), high.tolist()], 'grid_shape': shape,
            'supported_voxels': int(keep.sum()), 'sha256': sha(RUNS/'initialization.npz')}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-cross-temporal-static', action='store_true')
    args = parser.parse_args()
    if not args.enable_cross_temporal_static:
        raise SystemExit('Cross-temporal reconstruction is opt-in')
    assert not (REPORT/'discovery.json').exists(), 'Preserve completed discovery'
    cv2.setNumThreads(2); cv2.setRNGSeed(20261103)
    protocol = read(REPORT/'protocol.json'); meta = read(DATA/'transforms_train.json')
    frames = sorted(meta['frames'], key=lambda f: int(Path(f['file_path']).stem))
    images = [rgb(f, 400) for f in frames]
    ids = [int(Path(f['file_path']).stem) for f in frames]
    views = {n: w2c(f['transform_matrix'], 'metadata_cv') for n, f in zip(ids, frames)}
    k = intrinsic(meta['camera_angle_x'], 'metadata', 800, 450)
    sift = cv2.SIFT_create(nfeatures=protocol['sift_features'])
    features, descriptors = [], []
    for f in frames:
        image = (rgb(f, 800)*255).round().astype('uint8')
        kp, d = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
        features.append(np.asarray([p.pt for p in kp])); descriptors.append(d)
    edges, pair_counts = [], []
    for i in range(len(frames)):
        for offset in protocol['pair_offsets']:
            j = i+offset
            if j >= len(frames): continue
            pairs = mutual_matches(descriptors[i], descriptors[j], protocol['matching_ratio'])
            if not pairs: continue
            ii, jj = np.asarray(pairs).T
            error = epipolar_errors(features[i][ii], features[j][jj], views[ids[i]], views[ids[j]], k)
            good = error <= protocol['epipolar_limit_px']
            edges.extend([((i, int(a)), (j, int(b))) for a, b in np.asarray(pairs)[good]])
            pair_counts.append([ids[i], ids[j], len(pairs), int(good.sum())])
        if i % 20 == 0: print('CROSS_TEMPORAL_MATCH', i, 'of', len(frames), 'edges', len(edges), flush=True)
    linked, conflicts = join_tracks(edges); tracks = []
    for linked_track in linked:
        if len(linked_track) < 2: continue
        track_ids = [ids[i] for i, _ in linked_track]
        uv = np.asarray([features[i][j] for i, j in linked_track])
        status = evidence(track_ids, uv, views, k, **protocol['track_evidence'])
        tracks.append({**status, 'source': 'SIFT', 'frames': track_ids, 'uv': (uv/2).tolist()})
    # Reuse only prior train-RGB feature evidence. No historical GS model is read.
    old = read(ROOT/'experiments/staged_reconstruction/S1_automatic/discovery.json')
    for path, expected in old['training_image_sha256'].items(): assert sha(DATA/(path+'.png')) == expected
    tracks += [{**r, 'source': 'LK'} for r in old['tracks']]
    anchors, occupied = [], set()
    for r in sorted(tracks, key=lambda r: -r['observations']):
        if r['label'] != 'static_candidate' or r['frame_span'] < protocol['minimum_source_span']: continue
        key = tuple(np.rint(np.asarray(r['xyz'])/.025).astype(int))
        if key in occupied: continue
        color = np.median([pixel_colors(images[ids.index(n)], np.asarray([uv]))[0] for n, uv in zip(r['frames'], r['uv'])], 0)
        if color.mean() <= .025: continue  # untextured black cannot supply finite geometry
        occupied.add(key); anchors.append({**r, 'rgb': color.tolist()})
    assert len(anchors) >= 12, 'Insufficient cross-temporal anchors'
    incompatible = [r for r in tracks if r['label'] == 'incompatible_fixed_point']
    RUNS.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    initial, geometry, panels = {}, {}, []
    for n, f, image in zip(ids, frames, images):
        a, b = region_confidence(f, image, anchors, incompatible, meta, protocol)
        initial[str(n)] = a; geometry[str(n)] = b
        if n in [2, 27, 43, 86, 166, 188, 200]:
            panel = Image.new('RGB', (1200, 250), '#161922')
            for column, (mask, title) in enumerate([(None, 'Observed RGB'), (a, 'Initial confidence'), (b, 'Geometric candidate upper bound')]):
                overlay = (image*255).round().astype('uint8')
                if mask is not None:
                    pos = mask >= .5; overlay[pos] = .35*overlay[pos]+.65*np.array([0, 255, 100])
                panel.paste(Image.fromarray(overlay), (400*column, 25))
                ImageDraw.Draw(panel).text((400*column+8, 5), f't{n} {title}', fill='white')
            panels.append(np.asarray(panel))
    np.savez_compressed(RUNS/'initial_confidence.npz', **initial)
    np.savez_compressed(RUNS/'geometric_confidence.npz', **geometry)
    Image.fromarray(np.concatenate(panels)).save(OUT/'confidence_proposals.png')
    init = initialize(anchors, geometry, frames, images, meta, protocol)
    write(REPORT/'discovery.json', {'SIFT_counts': dict(Counter(r['label'] for r in tracks if r['source'] == 'SIFT')),
        'LK_counts': old['counts'], 'deduplicated_anchors': len(anchors), 'anchor_sources': dict(Counter(a['source'] for a in anchors)),
        'same_frame_merge_conflicts_rejected': conflicts, 'pairs': pair_counts, 'anchors': anchors,
        'incompatible': incompatible, 'initial': summary(initial), 'candidate_upper_bound': summary(geometry), 'initialization': init,
        'known_static_semantics_used': False, 'independent_geometry_GT': False,
        'training_image_sha256': old['training_image_sha256'], 'metadata_sha256': sha(DATA/'transforms_train.json'),
        'reused_LK_evidence_sha256': sha(ROOT/'experiments/staged_reconstruction/S1_automatic/discovery.json')})
    print('CROSS_TEMPORAL_PREPARED', len(anchors), 'anchors', init['static_points'], 'Gaussians',
          summary(initial)['fraction_all_image_pixels'], summary(geometry)['fraction_all_image_pixels'], flush=True)


if __name__ == '__main__': main()
