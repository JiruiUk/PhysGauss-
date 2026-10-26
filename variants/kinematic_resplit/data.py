"""Immutable, time-disjoint 120/30/50 split and physically separate loader views."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False)+'\n'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError('Immutable data manifest changed: '+str(path))
    path.write_text(text, encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def frame_id(frame):
    return int(Path(frame['file_path'].replace('\\', '/')).stem)


def audit_original(source):
    source = Path(source)
    original = {s: read(source/f'transforms_{s}.json')['frames'] for s in ('train', 'test')}
    ids = {s: {frame_id(f) for f in fs} for s, fs in original.items()}
    times = {s: {f['time'] for f in fs} for s, fs in original.items()}
    independent = (source/'transforms_val.json').exists()
    return {'counts': {s: len(fs) for s, fs in original.items()},
            'independent_validation_exists': independent,
            'validation_test_role_overlap': len(ids['test']) if not independent else None,
            'train_test_image_overlap': sorted(ids['train'] & ids['test']),
            'train_test_equal_metadata_times': sorted(times['train'] & times['test']),
            'note': 'Historical checkpoint selection calls getTestCameras(); validation is the test set.'}


def partition(frames):
    ordered = sorted(copy.deepcopy(frames), key=frame_id)
    if [frame_id(f) for f in ordered] != list(range(1, 201)):
        raise ValueError('Expected exactly one acquisition per original frame ID 1..200')
    test = [f for f in ordered if (frame_id(f)-1) % 4 == 0]
    available = [f for f in ordered if (frame_id(f)-1) % 4 != 0]
    # Middle member of each 5-frame block in the original training sequence.
    # Fixed before any new training/validation result; 30 spread-out timestamps.
    val_ids = {frame_id(available[i]) for i in range(2, 150, 5)}
    parts = {'train': [f for f in available if frame_id(f) not in val_ids],
             'val': [f for f in available if frame_id(f) in val_ids], 'test': test}
    for fs in parts.values():
        for f in fs:
            f['file_path'] = f['file_path'].replace('\\', '/')
            f['time'] = (frame_id(f)-1)/199
    return parts


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not os.path.samefile(source, target):
            if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
                raise ValueError('Dataset image changed: '+str(target))
        return
    try: os.link(source, target)
    except OSError: shutil.copyfile(source, target)


def random_ply(path, count):
    import numpy as np
    from plyfile import PlyData, PlyElement
    if path.exists(): return
    rng = np.random.default_rng(0)
    vertices = np.zeros(count, dtype=[(n, 'f4') for n in ('x', 'y', 'z', 'nx', 'ny', 'nz')]+
                        [(n, 'u1') for n in ('red', 'green', 'blue')])
    xyz = rng.uniform(-1.3, 1.3, (count, 3)).astype('float32')
    for j, name in enumerate(('x', 'y', 'z')): vertices[name] = xyz[:, j]
    for name in ('red', 'green', 'blue'): vertices[name] = 127
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(str(path))


def materialize(source, destination, header, parts, point_count=100000):
    source, destination = Path(source), Path(destination)
    for mode, heldout in [('training', 'val'), ('final_test', 'test')]:
        view = destination/mode
        view.mkdir(parents=True, exist_ok=True)
        write(view/'transforms_train.json', dict(header, frames=parts['train']))
        # Legacy loader slot; its semantic role is explicit, never a merged pool.
        write(view/'transforms_test.json', dict(header, frames=parts[heldout]))
        write(view/'loader_role.json', {'test_slot_semantics': heldout, 'train_count': len(parts['train']),
                                      'heldout_count': len(parts[heldout]), 'point_count': point_count})
        for f in parts['train']+parts[heldout]:
            relative = f['file_path']+'.png'
            link(source/relative, view/relative)
        # Flow may have been estimated using held-out neighboring RGB. It is not copied.
        # Depth provenance is also not assumed. The current model uses neither.
        for f in parts['train']:
            relative = 'mask/'+Path(f['file_path']).name+'.png'
            if (source/relative).exists(): link(source/relative, view/relative)
        random_ply(view/'points3d.ply', point_count)
    for s, fs in parts.items(): write(destination/f'transforms_{s}.json', dict(header, frames=fs))


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    original = read(source/'transforms_all.json')
    parts = partition(original['frames'])
    actual_test = {frame_id(f) for f in read(source/'transforms_test.json')['frames']}
    if actual_test != {frame_id(f) for f in parts['test']}:
        raise ValueError('Official test membership changed')
    header = {k: v for k, v in original.items() if k != 'frames'}
    materialize(source, destination, header, parts)
    manifest = {'source': str(source.resolve()), 'destination': str(destination.resolve()),
        'original_audit': audit_original(source),
        'counts': {s: len(fs) for s, fs in parts.items()},
        'frame_ids': {s: [frame_id(f) for f in fs] for s, fs in parts.items()},
        'time_rule': '(original_frame_id - 1) / 199; uniform capture cadence assumed, not verified',
        'split_rule': 'official 50 test; every fifth original training frame, offset 2, becomes validation',
        'pairwise_image_and_time_overlap': 0,
        'task': 'same-scene temporal interpolation, not future extrapolation or contiguous-gap recovery',
        'point_cloud': '100000 independent uniform random points; seed 0; no released geometry or trained weights',
        'prior_exposure': 'The 50 official test images were used in previous studies; prospective separation does not erase historical tuning.',
        'original_metadata_sha256': {s: hashlib.sha256((source/f'transforms_{s}.json').read_bytes()).hexdigest()
                                    for s in ('all', 'train', 'test')}}
    write(destination/'split_manifest.json', manifest)
    return manifest
