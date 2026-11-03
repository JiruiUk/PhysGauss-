"""Initialize a geometry-supported static shell without known static masks."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from ..prepare_data import ROOT, DATA, read, write, sha
from output.short_clip_diagnostic.camera_math import w2c, intrinsic, project
from .discover import REPORT, RUNS, rgb


def verify():
    for path, expected in read(REPORT/'source_freeze.json').items():
        assert sha(ROOT/path) == expected, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-automatic-static', action='store_true')
    args = parser.parse_args()
    if not args.enable_automatic_static:
        raise SystemExit('Automatic static initialization is opt-in')
    assert not (RUNS/'initialization.npz').exists(), 'Preserve existing initialization'
    protocol = read(REPORT/'protocol.json')
    discovery = read(REPORT/'discovery.json')
    accepted = [r for r in discovery['tracks'] if r['label'] == 'static_candidate']
    assert len(accepted) >= 12, 'Insufficient static geometric evidence; no color fallback'
    xyz = np.asarray([r['xyz'] for r in accepted])
    low, high = xyz.min(0)-.16, xyz.max(0)+.16
    spacing = protocol['voxel_spacing']
    axes = [np.arange(a, b+spacing/2, spacing, dtype=np.float32) for a, b in zip(low, high)]
    shape = tuple(map(len, axes))
    assert np.prod(shape) <= 3000000, 'Unbounded triangulation; inspect evidence without silently clipping'
    points = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    near = cKDTree(xyz).query(points)[0] <= .12
    pos, seen = np.zeros(len(points), int), np.zeros(len(points), int)
    thirds = np.zeros((len(points), 3), int)
    colors = np.zeros((len(points), 3), float)
    meta = read(DATA/'transforms_train.json')
    masks = np.load(RUNS/'inferred_masks.npz')
    k = intrinsic(meta['camera_angle_x'], 'metadata', 400, 225)
    panels = []
    for frame in meta['frames']:
        number = int(Path(frame['file_path']).stem)
        image = rgb(frame, 400)
        mask = masks[str(number)]
        uv, depth = project(points, w2c(frame['transform_matrix'], 'metadata_cv'), k)
        good = (depth > .2)&np.isfinite(uv).all(1)&(uv[:, 0] >= 0)&(uv[:, 0] < 400)&(uv[:, 1] >= 0)&(uv[:, 1] < 225)
        rows = np.flatnonzero(good)
        xy = uv[rows].astype(int)
        supported = mask[xy[:, 1], xy[:, 0]]
        seen[rows] += 1
        pos[rows] += supported
        thirds[rows, min((number-1)*3//200, 2)] += supported
        colors[rows] += image[xy[:, 1], xy[:, 0]]*supported[:, None]
        if number in [2, 27, 43, 86, 166, 188, 200]:
            original = (image*255).round().astype('uint8')
            overlay = original.copy()
            overlay[mask] = .3*overlay[mask]+.7*np.array([0, 255, 100])
            panel = Image.new('RGB', (800, 250), '#161922')
            panel.paste(Image.fromarray(original), (0, 25))
            panel.paste(Image.fromarray(overlay), (400, 25))
            ImageDraw.Draw(panel).text((8, 5), f'train {number}; RGB / inferred static (green), other pixels unknown', fill='white')
            panels.append(np.asarray(panel))
    keep = near&(pos >= 8)&(seen >= 12)&((thirds >= 2).sum(1) >= 2)
    volume = keep.reshape(shape)
    rows = np.flatnonzero((volume&~binary_erosion(volume)).reshape(-1))
    assert len(rows) >= 100, 'Insufficient supported static geometry; no semantic fallback'
    if len(rows) > protocol['max_static_points']:
        rows = np.sort(np.random.default_rng(protocol['seed']).choice(rows, protocol['max_static_points'], False))
    np.savez_compressed(RUNS/'initialization.npz', xyz=points[rows],
        rgb=(colors[rows]/np.maximum(pos[rows, None], 1)).astype(np.float32), spacing=np.asarray(spacing))
    out = ROOT/'output/staged_reconstruction/S1_automatic'
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(panels)).save(out/'inferred_masks.png')
    write(REPORT/'initialization.json', {'static_points': len(rows), 'accepted_tracks': len(accepted),
        'bounds': [low.tolist(), high.tolist()], 'grid_shape': shape, 'supported_voxels': int(keep.sum()),
        'initialization_sha256': sha(RUNS/'initialization.npz'), 'mask_sha256': sha(RUNS/'inferred_masks.npz'),
        'image_review_file': (out/'inferred_masks.png').relative_to(ROOT).as_posix(),
        'physical_geometry_GT_available': False, 'unknown_regions_are_not_empty': True})
    files = list(Path(__file__).parent.glob('*.py'))+[ROOT/'tests/test_automatic_static.py',
        REPORT/'protocol.json', REPORT/'discovery.json', RUNS/'initialization.npz', RUNS/'inferred_masks.npz']
    frozen = read(ROOT/'experiments/staged_reconstruction/S3_union/source_freeze.json')
    for path, expected in frozen.items():
        assert sha(ROOT/path) == expected, path
    frozen.update({p.relative_to(ROOT).as_posix(): sha(p) for p in files})
    write(REPORT/'source_freeze.json', frozen)
    print('AUTOMATIC_STATIC_INITIALIZED', len(rows), 'points from', len(accepted), 'geometric tracks', flush=True)


if __name__ == '__main__':
    main()
