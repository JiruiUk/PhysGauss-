"""CPU-only nested whole-frame dropout. Original RGB/flow/mask files are read-only.

Only training manifests change. Directory links avoid copying the large dataset.
The official test split is unchanged; dropped training frames form a second,
independent evaluation split and are never used for snapshot selection.
"""
from copy import deepcopy
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from pidg_benchmark import read_json, sha256, write_json

SOURCE = ROOT / 'data/PIDG/rubber_duck'
DATA = ROOT / 'data/pidg_random_frame_mask/rubber_duck'
REPORTS = ROOT / 'experiments/random_frame_mask'
RATIOS = (10, 30, 50, 70, 90)
MASK_SEED = 0
SOURCE_FILES = ('transforms_train.json', 'transforms_test.json', 'points3d.ply', 'audit.json', 'download_manifest.json')


def split_frames(train, test, percentages=RATIOS, seed=MASK_SEED):
    train_names = [frame['file_path'] for frame in train['frames']]
    test_names = [frame['file_path'] for frame in test['frames']]
    if len(set(train_names)) != len(train_names) or len(set(test_names)) != len(test_names):
        raise ValueError('Duplicate frame IDs')
    if set(train_names) & set(test_names):
        raise ValueError('Original train/test overlap')
    if not train_names or not test_names:
        raise ValueError('Empty source split')
    permutation = list(range(len(train_names)))
    random.Random(seed).shuffle(permutation)
    output = []
    for percent in percentages:
        if type(percent) is not int or not 0 < percent < 100:
            raise ValueError('Use integer dropout percentages between 1 and 99')
        count = round(len(train_names) * percent / 100)
        if not 0 < count < len(train_names):
            raise ValueError('Dropout must leave both retained and dropped frames')
        removed = set(permutation[:count])
        kept = [deepcopy(f) for i, f in enumerate(train['frames']) if i not in removed]
        dropped = [deepcopy(f) for i, f in enumerate(train['frames']) if i in removed]
        output.append({'id': f'drop{percent:02d}_seed{seed}', 'drop_percent': percent, 'mask_seed': seed,
                       'train': {**deepcopy(train), 'frames': kept},
                       'test': deepcopy(test), 'heldout': {**deepcopy(train), 'frames': dropped},
                       'retained_count': len(kept), 'dropped_count': len(dropped),
                       'retained_ids': [f['file_path'] for f in kept],
                       'dropped_ids': [f['file_path'] for f in dropped],
                       'retained_time_min': min(f['time'] for f in kept),
                       'retained_time_max': max(f['time'] for f in kept)})
    return output


def link_directory(destination, source):
    destination, source = Path(destination).absolute(), Path(source).resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source:
            raise ValueError('Existing data link points elsewhere: ' + str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        # Directory junctions require no symlink privilege. Never remove targets.
        subprocess.run([os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c', 'mklink', '/J',
                        str(destination), str(source)], check=True, stdout=subprocess.DEVNULL)
    else:
        destination.symlink_to(source, target_is_directory=True)
    if destination.resolve() != source:
        raise ValueError('Data link verification failed')


def save_frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Existing fixture differs: ' + str(path))
    else:
        write_json(path, value)


def prepare(source=SOURCE, output=DATA):
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    source_hashes = {name: sha256(source / name) for name in SOURCE_FILES}
    train, test = read_json(source / 'transforms_train.json'), read_json(source / 'transforms_test.json')
    records = split_frames(train, test)
    for item in records:
        common = output / item['id']
        for split, test_doc in [('dataset', item['test']), ('heldout_dataset', item['heldout'])]:
            directory = common / split
            directory.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                if child.is_dir():
                    link_directory(directory / child.name, child)
            target = directory / 'points3d.ply'
            if target.exists():
                if sha256(target) != source_hashes['points3d.ply']:
                    raise ValueError('Canonical point cloud changed')
            else:
                shutil.copyfile(source / 'points3d.ply', target)
            save_frozen_json(directory / 'transforms_train.json', item['train'])
            save_frozen_json(directory / 'transforms_test.json', test_doc)
            audit = {'scene': 'rubber_duck', 'train_count': item['retained_count'],
                     'test_count': len(test_doc['frames']), 'missing': [], 'shape_mismatches': [],
                     'drop_percent': item['drop_percent'], 'mask_seed': item['mask_seed'],
                     'original_source': str(source), 'role': split,
                     'note': 'Original times, poses, RGB and annotations preserved; manifests alone filter training.'}
            for frame in item['train']['frames'] + test_doc['frames']:
                image = directory / (frame['file_path'] + '.png')
                if not image.is_file():
                    raise FileNotFoundError(image)
            save_frozen_json(directory / 'audit.json', audit)
            save_frozen_json(directory / 'download_manifest.json', {'local_overlay': True, 'source_hashes': source_hashes})
        manifest = {k: v for k, v in item.items() if k not in ('train', 'test', 'heldout')}
        manifest.update(source=str(source), source_hashes=source_hashes,
                        original_train_count=len(train['frames']), original_test_count=len(test['frames']),
                        official_test_unchanged=True, timestamps_unchanged=True,
                        physics_train_count=item['retained_count'])
        save_frozen_json(common / 'split_manifest.json', manifest)
    if {name: sha256(source / name) for name in SOURCE_FILES} != source_hashes:
        raise RuntimeError('Original dataset changed during preparation')
    return records


def fixture_hashes(directory):
    return {name: sha256(Path(directory) / name) for name in SOURCE_FILES}


if __name__ == '__main__':
    for item in prepare():
        print(item['id'], 'retained', item['retained_count'], 'dropped', item['dropped_count'])
