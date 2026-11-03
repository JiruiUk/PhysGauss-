"""Unmasked RGB tracking and held-observation fixed-world-point discovery."""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb
from scipy.ndimage import binary_erosion
from ..prepare_data import ROOT, DATA, read, write, sha
from output.short_clip_diagnostic.camera_math import w2c, intrinsic
from .geometry import evidence, region_support

REPORT = ROOT/'experiments/staged_reconstruction/S1_automatic'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_automatic'


def rgb(frame, width):
    # Dataset rendering convention only. Alpha is discarded after black compositing;
    # it is never used as static/dynamic/foreground supervision.
    with Image.open(DATA/(frame['file_path']+'.png')) as image:
        rgba = np.asarray(image.convert('RGBA').resize((width, width*9//16), Image.Resampling.BILINEAR), dtype=np.float32)/255
    return rgba[..., :3]*rgba[..., 3:4]


def track(frames, images):
    active, complete = [], []
    settings = dict(winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, .01))
    previous = None
    for index, (frame, image) in enumerate(zip(frames, images)):
        number = int(Path(frame['file_path']).stem)
        gray = cv2.cvtColor((image*255).round().astype('uint8'), cv2.COLOR_RGB2GRAY)
        if active:
            p = np.float32([row[-1][1] for row in active]).reshape(-1, 1, 2)
            q, status, err = cv2.calcOpticalFlowPyrLK(previous, gray, p, None, **settings)
            back, bs, be = cv2.calcOpticalFlowPyrLK(gray, previous, q, None, **settings)
            good = (status[:, 0] > 0)&(bs[:, 0] > 0)&(err[:, 0] < 20)&(be[:, 0] < 20)
            good &= np.isfinite(q[:, 0]).all(1)&(np.linalg.norm(back[:, 0]-p[:, 0], axis=1) < .5)
            good &= (q[:, 0, 0] >= 0)&(q[:, 0, 0] < gray.shape[1])&(q[:, 0, 1] >= 0)&(q[:, 0, 1] < gray.shape[0])
            keep = []
            for row, point, valid in zip(active, q[:, 0], good):
                if valid:
                    keep.append(row+[(number, point.astype(float))])
                else:
                    complete.append(row)
            active = keep
        if index % 4 == 0:
            mask = np.full(gray.shape, 255, dtype=np.uint8)
            for row in active:
                cv2.circle(mask, tuple(np.rint(row[-1][1]).astype(int)), 4, 0, -1)
            found = cv2.goodFeaturesToTrack(gray, 1000, .003, 3, mask=mask)
            if found is not None:
                active.extend([[(number, point.astype(float))] for point in found[:, 0]])
        previous = gray
    return complete+active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-automatic-static', action='store_true')
    args = parser.parse_args()
    if not args.enable_automatic_static:
        raise SystemExit('Automatic static discovery is opt-in; pass --enable-automatic-static')
    if (REPORT/'discovery.json').exists():
        raise SystemExit('Preserve existing discovery; no overwrite or automatic threshold search')
    cv2.setNumThreads(2)
    protocol = read(REPORT/'protocol.json')
    meta = read(DATA/'transforms_train.json')
    frames = sorted(meta['frames'], key=lambda f: int(Path(f['file_path']).stem))
    images = [rgb(frame, 400) for frame in frames]
    views = {int(Path(f['file_path']).stem): w2c(f['transform_matrix'], 'metadata_cv') for f in frames}
    k = intrinsic(meta['camera_angle_x'], 'metadata', 400, 225)
    tracks = []
    for row in track(frames, images):
        if len(row) < 2:
            continue
        ids, uv = zip(*row)
        status = evidence(ids, uv, views, k, **protocol['track_evidence'])
        tracks.append({**status, 'frames': list(ids), 'uv': [p.tolist() for p in uv]})
    counts = Counter(row['label'] for row in tracks)
    RUNS.mkdir(parents=True, exist_ok=True)
    masks, colors = {}, []
    for frame, image in zip(frames, images):
        number = int(Path(frame['file_path']).stem)
        segments = felzenszwalb(image, scale=100, sigma=.8, min_size=60)
        seeds = {name: [] for name in ['static_candidate', 'incompatible_fixed_point']}
        for row in tracks:
            if row['label'] in seeds and number in row['frames']:
                seeds[row['label']].append(row['uv'][row['frames'].index(number)])
        mask = region_support(segments, seeds['static_candidate'], seeds['incompatible_fixed_point'])
        masks[str(number)] = binary_erosion(mask, iterations=1)
    np.savez_compressed(RUNS/'inferred_masks.npz', **masks)
    result = {'input_contract': protocol['input_contract'], 'camera_contract': 'metadata_cv',
        'semantic_static_labels_used': False, 'known_static_colors_used': False,
        'S1_weights_or_old_tracks_used': False, 'counts': dict(counts), 'tracks': tracks,
        'mask_pixels_per_frame': {n: int(v.sum()) for n, v in masks.items()},
        'mask_sha256': sha(RUNS/'inferred_masks.npz'),
        'train_metadata_sha256': sha(DATA/'transforms_train.json'),
        'training_image_sha256': {f['file_path']: sha(DATA/(f['file_path']+'.png')) for f in frames},
        'limits': ['Stationarity is conditional on camera calibration, correct tracking and observed intervals',
                   'Incompatible tracks may include tracking error; no physical body/material labels',
                   'Region propagation is an appearance-smoothness hypothesis, not static or geometric GT',
                   'No static discovery is possible for wholly unobserved/textureless ambiguous regions']}
    write(REPORT/'discovery.json', result)
    print('AUTOMATIC_STATIC_DISCOVERY', dict(counts), 'positive_pixels',
          min(result['mask_pixels_per_frame'].values()), max(result['mask_pixels_per_frame'].values()), flush=True)


if __name__ == '__main__':
    main()
