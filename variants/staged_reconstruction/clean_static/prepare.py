"""Train-only visibility audit. Retains full parent and quarantined hypotheses."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
import torch
from output.short_clip_diagnostic.camera_math import w2c, intrinsic
from ..prepare_data import ROOT, DATA, read, write, sha
from ..automatic_static.discover import rgb
from ..adaptive_static.prepare import REPORT as PREVIOUS_REPORT, RUNS as PREVIOUS_RUNS, verify as verify_previous, world_points
from .evidence import accumulation

REPORT = ROOT/'experiments/staged_reconstruction/S1_clean_visibility/v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_clean_visibility/v1'
OUT = ROOT/'output/staged_reconstruction/S1_clean_visibility/v1'


def verify():
    verify_previous()
    for path, expected in read(REPORT/'source_freeze.json').items(): assert sha(ROOT/path) == expected, path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-clean-static', action='store_true'); args = parser.parse_args()
    if not args.enable_clean_static: raise SystemExit('Clean static reconstruction is opt-in')
    verify_previous(); assert not (REPORT/'discovery.json').exists(), 'Keep prior discovery'
    assert not (RUNS/'stop.request').exists(), 'Saved user pause; do not resume'
    protocol = read(REPORT/'protocol.json'); path = ROOT/protocol['parent']; assert sha(path) == protocol['parent_sha256']
    state = torch.load(path, map_location='cpu', weights_only=False)
    meta = read(DATA/'transforms_train.json')
    frames = sorted(meta['frames'], key=lambda f: int(Path(f['file_path']).stem)); ids = [int(Path(f['file_path']).stem) for f in frames]
    views = np.asarray([w2c(f['transform_matrix'], 'metadata_cv') for f in frames]); k = intrinsic(meta['camera_angle_x'], 'metadata', 400, 225)
    with np.load(PREVIOUS_RUNS/'verified_depths.npz') as data: depths = np.asarray([data[str(n)] for n in ids])
    with np.load(PREVIOUS_RUNS/'geometric_confidence.npz') as data: geometry = np.asarray([data[str(n)] for n in ids])
    with np.load(PREVIOUS_RUNS/'reservoir.npz') as data: reservoir = {n: data[n].copy() for n in data.files}
    reliable = geometry >= protocol['reliable_geometry_min']
    report = {}
    for label, xyz in [('parent', state['model']['_xyz'].numpy()), ('reservoir', reservoir['xyz'])]:
        result = accumulation(xyz, views, k, depths, reliable, protocol, ids)
        np.savez_compressed(RUNS/f'{label}_evidence.npz', **result)
        report[label] = {'points': len(xyz), 'quarantined': int(result['quarantine'].sum()),
            'retained': int((~result['quarantine']).sum()), 'has_positive_geometry': int((result['positive'] > 0).sum()),
            'all_unknown_or_occluded': int(((result['positive']+result['free']) == 0).sum()),
            'quarantined_positive_median': float(np.median(result['positive'][result['quarantine']])) if result['quarantine'].any() else None,
            'quarantined_free_median': float(np.median(result['free'][result['quarantine']])) if result['quarantine'].any() else None}
    rejected = result['quarantine']; lookup = cKDTree(reservoir['xyz'])
    filtered, counts, panels = {}, {}, []
    for i, (frame, number) in enumerate(zip(frames, ids)):
        value = geometry[i].copy(); yy, xx = np.nonzero(depths[i] > 0)
        xyz = world_points(np.c_[xx, yy], depths[i][yy, xx], views[i], k)
        distance, index = lookup.query(xyz)
        bad = rejected[index] & (distance <= .05)
        value[yy[bad], xx[bad]] = 0
        filtered[str(number)] = value
        counts[str(number)] = [int((geometry[i] >= .5).sum()), int((value >= .5).sum())]
        if number in [2,27,43,86,166,188,200]:
            panel = Image.new('RGB', (800,250), '#161922')
            for col, (m, title) in enumerate([(geometry[i] >= .5, 'Original geometry'), (value >= .5, 'Visibility-filtered; excluded = unknown')]):
                im = (rgb(frame, 400)*255).round().astype(np.uint8); im[m] = .35*im[m]+.65*np.array([0,255,100])
                panel.paste(Image.fromarray(im), (col*400,25)); ImageDraw.Draw(panel).text((col*400+5,5), f't{number} {title}', fill='white')
            panels.append(np.asarray(panel))
    np.savez_compressed(RUNS/'geometric_confidence.npz', **filtered)
    Image.fromarray(np.concatenate(panels)).save(OUT/'candidate_filter.png')
    write(REPORT/'discovery.json', {'parent_sha256': sha(path), 'protocol_sha256': sha(REPORT/'protocol.json'),
        'evidence': report, 'pixels_before_after': counts,
        'all_pixels_before': sum(v[0] for v in counts.values()), 'all_pixels_after': sum(v[1] for v in counts.values()),
        'frames_retained': sum(v[1] > 0 for v in counts.values()), 'independent_scene_GT': False,
        'quarantined_is_unknown_not_dynamic_or_empty': True,
        'inputs': {str(p.relative_to(ROOT)).replace('\\','/'): sha(p) for p in [path, PREVIOUS_RUNS/'verified_depths.npz', PREVIOUS_RUNS/'geometric_confidence.npz', PREVIOUS_RUNS/'reservoir.npz', REPORT/'protocol.json']},
        'artifacts': {p.name: sha(p) for p in RUNS.glob('*.npz')}})
    print('CLEAN_STATIC_PREPARED', report, 'pixels', sum(v[0] for v in counts.values()), sum(v[1] for v in counts.values()), flush=True)


if __name__ == '__main__': main()
