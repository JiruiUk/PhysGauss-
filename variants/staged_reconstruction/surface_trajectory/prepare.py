"""R0 train-only observation audit. Explicit opt-in; no model optimizer."""
import argparse
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from ..prepare_data import ROOT, REPORT, RUNS, DATA, read, write, sha
from ..union.core import dynamic_mask
from .observations import extract_window, balanced_subset, appearance_masks

OUT = REPORT/'surface_trajectory'
PARENT = RUNS/'S3_union/physics/formal/checkpoint.pt'
PARENT_SHA = '8c767b7835eee936e49989369d7bc0277d7567d3aedaa7385c06654284b68ad0'
CONFIG = dict(version=1, seed=20261103, min_observations=8, max_observations=16,
    source_stride=8, max_corners=64, corner_quality=.01, seed_distance=4.,
    patch_radius=2, patch_std_min=2., ncc_min=.6, fb_max_px=.75,
    fragment_closure_max_px=.75, mask_erosion=2, candidate_target=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-surface-trajectory', action='store_true')
    parser.add_argument('--observation-version', type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    if not args.enable_surface_trajectory:
        raise SystemExit('Surface trajectory preparation is opt-in')
    config = dict(CONFIG)
    out = OUT
    if args.observation_version >= 2:
        config.update(version=args.observation_version, source_stride=2, check_valid_prefixes=True)
        out = OUT/f'observation_v{args.observation_version}'
    if args.observation_version == 3:
        config.update(mask_policy='yellow_orange_holefilled_envelope', suppress_highlights=True,
            highlight_policy='minRG>.65,B>.5,B>.65minRG; forbid5x5neighborhood at every visible observation')
    cv2.setNumThreads(2)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out/'observation_protocol.json'
    if config_path.exists():
        assert read(config_path) == config, 'Do not overwrite a frozen observation protocol'
    else:
        write(config_path, config)
    if (out/'image_tracks.json').exists():
        raise SystemExit('Preserve existing observation result; audit before changing versions')
    assert sha(PARENT) == PARENT_SHA
    meta_path = DATA/'transforms_train.json'
    meta = read(meta_path)
    frames = sorted(meta['frames'], key=lambda f: int(Path(f['file_path']).stem))
    ids = [int(Path(f['file_path']).stem) for f in frames]
    times = [float(f['time']) for f in frames]
    cache = ROOT/'data/derived/flow_comparison_train_only'
    manifest_path = cache/'manifest.json'
    manifest = read(manifest_path)
    assert ids == manifest['train_frame_ids'] and len(ids) == 120
    assert not manifest['heldout_RGB_used']
    lock = manifest['model_lock']
    assert sha(Path(lock['weights_path'])) == lock['weights_sha256']
    image_hashes, grays, masks = {}, [], []
    for frame in frames:
        path = DATA/(frame['file_path']+'.png')
        digest = sha(path)
        relative = str(path.relative_to(ROOT))
        assert digest == manifest['input_sha256'][relative]
        image_hashes[relative] = digest
        a = np.asarray(Image.open(path).convert('RGBA').resize((400, 225), Image.Resampling.BILINEAR))
        grays.append(cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32))
        if config.get('suppress_highlights'):
            _, valid = appearance_masks(a.astype(np.float32)/255, config['mask_erosion'], True)
            masks.append(valid.astype(np.float32))
        else:
            masks.append(binary_erosion(dynamic_mask(a.astype(np.float32)/255), iterations=config['mask_erosion']).astype(np.float32))
    flows, flow_hashes = [], {}
    for index, pair in enumerate(manifest['pairs']):
        assert (pair['source'], pair['target']) == (ids[index], ids[index+1])
        path = cache/pair['file']
        digest = sha(path)
        assert digest == manifest['output_sha256'][pair['file']]
        with np.load(path, allow_pickle=False) as value:
            # Raw RGB-derived fields only: never read cached depth/motion/valid.
            forward, backward = value['forward'], value['backward']
            assert forward.shape == backward.shape == (225, 400, 2)
            assert np.isfinite(forward).all() and np.isfinite(backward).all()
            flows.append((forward, backward))
        flow_hashes[pair['file']] = digest
    tracks, windows = [], []
    for start in range(0, len(ids)-config['min_observations']+1, config['source_stride']):
        found, counts = extract_window(start, ids, times, grays, masks, flows, config)
        tracks.extend(found)
        windows.append({'source_frame': ids[start], **counts, 'qualified': len(found)})
        print('SURFACE_OBSERVATIONS', ids[start], counts, 'qualified', len(found), flush=True)
    selected = balanced_subset(tracks, config['candidate_target'], config['seed'])
    result = dict(stage='R0', training_only=True, image_space=[400, 225], tracks=selected,
        qualified_before_balancing=len(tracks), windows=windows,
        counts_by_time_band=dict(Counter(t['time_band'] for t in selected)),
        parent_sha256=PARENT_SHA, metadata_sha256=sha(meta_path),
        observation_protocol_sha256=sha(config_path), image_sha256=image_hashes,
        raw_flow_pair_sha256=flow_hashes, flow_manifest_sha256=sha(manifest_path),
        flow_weights_sha256=lock['weights_sha256'], flow_commit=lock['commit'],
        consumed_cache_fields=['forward', 'backward'],
        cached_depth_motion_valid_used=False, body_count_input_or_estimation=False,
        image_identity_or_material_GT=False, human_review_completed=False,
        optimizer_updates=0, binding_pending=True,
        limitation='Estimated image evidence; FB/appearance agreement is not material identity or occlusion re-ID')
    write(out/'image_tracks.json', result)
    audit = {}
    all_times = []
    for split in ['train', 'val', 'test']:
        p = ROOT/f'data/derived/kinematic_resplit/transforms_{split}.json'
        values = [f['time'] for f in read(p)['frames']]
        all_times.extend(values)
        audit[split] = dict(count=len(values), unique_times=len(set(values)),
                            maximum_views_per_time=max(Counter(values).values()), metadata_sha256=sha(p))
    audit['combined'] = dict(count=len(all_times), unique_times=len(set(all_times)),
                             maximum_views_per_time=max(Counter(all_times).values()))
    audit['independent_original_scene_geometry_available'] = False
    audit['geometry_note'] = 'No audited persistent material IDs, dynamic depth GT or synchronized views admitted; existing synthetic folders alone do not establish these.'
    write(out/'data_audit.json', audit)
    write(out/'source_freeze.json', {p.relative_to(ROOT).as_posix(): sha(p)
        for p in list(Path(__file__).parent.glob('*.py'))+[ROOT/'tests/test_surface_trajectory_observations.py']})
    print('SURFACE_OBSERVATION_COMPLETE', len(selected), 'of', len(tracks), result['counts_by_time_band'], flush=True)


if __name__ == '__main__':
    main()
