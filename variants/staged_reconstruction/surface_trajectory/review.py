"""GT-image crop gallery and fixed split. Does not claim visual review happened."""
import argparse
from collections import Counter
import html
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..prepare_data import ROOT, read, write, sha
from ..stage2.train import image
from .prepare import OUT
from .split import group_tracks, split_groups

CONFIG = dict(source_patch_diameter_px=4., duplicate_minimum_common_frames=3,
              duplicate_median_px=2., duplicate_maximum_px=3., weighted_support_overlap=.5,
              split_seed=20261103, minimum_holdout_groups=20, minimum_training_tracks=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-surface-trajectory', action='store_true')
    parser.add_argument('--observation-version', type=int, choices=[2, 3], default=2)
    args = parser.parse_args()
    if not args.enable_surface_trajectory:
        raise SystemExit('Surface review preparation is opt-in')
    out = OUT if args.observation_version == 2 else OUT/'landmark_v3'
    tracks_path = OUT/f'observation_v{args.observation_version}/image_tracks.json'
    binding_path = out/'bindings.json'
    tracks = read(tracks_path)['tracks']
    bindings = read(binding_path)['bindings']
    groups, edges = group_tracks(tracks, bindings, CONFIG)
    result = split_groups(groups, bindings, CONFIG['split_seed'], CONFIG['minimum_holdout_groups'], CONFIG['minimum_training_tracks'])
    result.update(image_tracks_sha256=sha(tracks_path), binding_sha256=sha(binding_path), grouping_protocol=CONFIG,
                  all_candidate_groups=groups, grouping_edges=edges,
                  track_loss_holdout_not_image_or_geometry_GT=True)
    for split in ['training', 'holdout']:
        ids = set(result.get(split+'_track_ids', []))
        result[split+'_time_bands'] = dict(Counter(t['time_band'] for t in bindings if t['track_id'] in ids))
    path = out/'split_manifest.json'
    if path.exists():
        assert read(path) == __import__('json').loads(__import__('json').dumps(result)), 'Preserve frozen split'
    else:
        write(path, result)
    destination = ROOT/'output/staged_reconstruction/surface_trajectory/observations'
    if args.observation_version == 3:
        destination = destination.with_name('landmark_v3')
    destination.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
    accepted = {b['track_id'] for b in bindings}
    source_frames = read(ROOT/'data/derived/kinematic_resplit/transforms_train.json')['frames']
    byid = {int(f['file_path'].split('/')[-1]): f for f in source_frames}
    cache = {}
    rows = {}
    for t in tracks:
        selected = [0, 1, 2]+np.linspace(3, len(t['frames'])-1, 5).round().astype(int).tolist()
        row = Image.new('RGB', (1024, 150), '#1b1e25')
        draw = ImageDraw.Draw(row)
        role = ('binding only; split not admitted' if t['track_id'] in accepted and not result['admitted']
                else 'train' if t['track_id'] in result.get('training_track_ids', [])
                else 'holdout' if t['track_id'] in accepted else 'rejected binding')
        draw.text((6, 4), f'Track {t["track_id"]} | {role} | estimated image correspondence, material identity unknown', font=font, fill='white')
        for column, index in enumerate(selected):
            n = t['frames'][index]
            if n not in cache:
                rgba = image(byid[n], 1600)
                cache[n] = Image.fromarray((rgba[..., :3]*rgba[..., 3:4]*255).round().astype(np.uint8))
            u, v = t['uv_400'][index]
            x, y = (u+.5)*4-.5, (v+.5)*4-.5
            crop = cache[n].transform((112, 112), Image.Transform.AFFINE,
                (64/112, 0, x-32, 0, 64/112, y-32), Image.Resampling.BILINEAR)
            d = ImageDraw.Draw(crop)
            for first, last in [((48, 56), (53, 56)), ((59, 56), (64, 56)), ((56, 48), (56, 53)), ((56, 59), (56, 64))]:
                d.line((first, last), fill='#ff2255', width=1)
            row.paste(crop, (column*128+8, 34))
            draw.text((column*128+8, 20), f'{n} '+('bootstrap' if index < 3 else 'future'), font=font, fill='white')
        file = destination/f'track_{t["track_id"]:03d}.png'
        row.save(file)
        rows[t['track_id']] = row
    review_ids = []
    for band in range(3):
        band_tracks = sorted([t for t in bindings if t['time_band'] == band], key=lambda t:(t['source_frame'], t['track_id']))
        picks = np.linspace(0, len(band_tracks)-1, min(8, len(band_tracks))).round().astype(int) if band_tracks else []
        chosen = [band_tracks[i]['track_id'] for i in picks]
        review_ids.extend(chosen)
        if chosen:
            sheet = Image.new('RGB', (1024, 150*len(chosen)))
            for j, track_id in enumerate(chosen):
                sheet.paste(rows[track_id], (0, j*150))
            sheet.save(destination/f'review_band_{band}.png')
    body = ['<!doctype html><meta charset="utf-8"><title>S3-U surface trajectory observations</title>',
            '<style>body{background:#171b23;color:white;font:16px sans-serif}img{max-width:100%;display:block;margin:12px 0}</style>',
            '<h1>Estimated image tracks and bootstrap bindings</h1>',
            '<p>Training images only. No physical material IDs or cross-occlusion identity claim. Crops use native images.</p>']
    body += [f'<img loading="lazy" src="track_{t["track_id"]:03d}.png" alt="track {t["track_id"]}">' for t in tracks]
    (destination/'index.html').write_text('\n'.join(body), encoding='utf-8')
    write(out/'review_candidates.json', dict(track_ids=review_ids, gallery=destination.relative_to(ROOT).as_posix(),
        images_generated=True, actual_visual_inspection_completed=False,
        independent_human_annotations_available=False, optimizer_updates=0))
    print('SURFACE_SPLIT', {k:v for k, v in result.items() if k in ['admitted','reason','training_time_bands','holdout_time_bands','actual_training_fraction']}, flush=True)
    print('GROUPS', len(result['groups']), 'train', len(result.get('training_track_ids', [])),
          'holdout', len(result.get('holdout_track_ids', [])), 'holdout_groups', len(result.get('holdout_groups', [])), flush=True)


if __name__ == '__main__':
    main()
