"""Opt-in clocks; original metadata and source files are never rewritten."""
import json
from pathlib import Path


def physics_time(fid, train_count, shared=False):
    return fid.float() if shared else fid.float()/train_count


def build_timeline(source):
    frames = {}
    for split in ('train', 'test'):
        metadata = json.loads((Path(source)/f'transforms_{split}.json').read_text(encoding='utf-8'))
        for frame in metadata['frames']:
            name = Path(frame['file_path']).stem
            if not name.isdigit() or name in frames:
                raise ValueError('Unique numeric frame filenames required')
            frames[name] = {'split': split, 'frame': int(name), 'metadata_time': frame['time']}
    ids = [r['frame'] for r in frames.values()]
    lo, hi = min(ids), max(ids)
    if len(ids) != len(set(ids)) or set(ids) != set(range(lo, hi+1)) or lo == hi:
        raise ValueError('A contiguous unique frame sequence is required')
    for row in frames.values(): row['time'] = (row['frame']-lo)/(hi-lo)
    return {'protocol': 'frame_index', 'minimum_frame': lo, 'maximum_frame': hi,
            'assumption': 'numeric filenames order uniformly spaced simulation frames; physical seconds unknown',
            'frames': frames}


def apply_timeline(scene, timeline):
    for bank in (scene.train_cameras, scene.test_cameras):
        for views in bank.values():
            for view in views:
                name = Path(view.image_name).stem
                view.fid = view.fid.new_tensor([timeline['frames'][name]['time']])


def configure_scene(scene, config, source):
    if config.get('enabled') and config.get('timeline', 'official_metadata') == 'frame_index':
        timeline = build_timeline(source)
        apply_timeline(scene, timeline)
        return timeline
    return None


def configure_scene_from_env(scene, source):
    import os
    path = os.environ.get('PIDG_METRIC_CONFIG')
    return configure_scene(scene, json.loads(Path(path).read_text(encoding='utf-8')) if path else {}, source)
