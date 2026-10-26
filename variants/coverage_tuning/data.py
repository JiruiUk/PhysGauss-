"""Read original RGBA alpha explicitly; never substitute a motion/SAM mask."""
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def frame_paths(source, split):
    source = Path(source).resolve()
    frames = json.loads((source/f'transforms_{split}.json').read_text())['frames']
    result = {}
    for f in frames:
        path = (source/(f['file_path']+'.png')).resolve()
        if not path.is_relative_to(source):
            raise ValueError('Image escapes the declared dataset view')
        name = Path(f['file_path']).stem
        if name in result: raise ValueError('Duplicate frame')
        result[name] = path
    return result


def read_alpha(path, size):
    with Image.open(path) as image:
        if image.mode != 'RGBA':
            raise ValueError('Original PNG must contain an explicit RGBA alpha channel')
        alpha = image.getchannel('A').resize(size, Image.Resampling.BICUBIC)
        array = np.array(alpha, dtype=np.float32)/255.
    return torch.from_numpy(array.copy())[None]


def audit_training(source):
    source = Path(source)
    train, val = frame_paths(source, 'train'), frame_paths(source, 'test')
    if set(train) & set(val): raise ValueError('Train/validation frame overlap')
    records = []
    for name, path in train.items():
        with Image.open(path) as image:
            if image.mode != 'RGBA': raise ValueError('Missing RGBA alpha: '+str(path))
            alpha = np.array(image.getchannel('A'))
            lo, hi = int(alpha.min()), int(alpha.max())
            fraction = float((alpha > 127).mean())
            if lo != 0 or hi != 255 or not .001 < fraction < .99:
                raise ValueError('Degenerate foreground alpha: '+str(path))
            records.append({'id': name, 'path': str(path), 'size': list(image.size),
                            'alpha_range': [lo, hi], 'foreground_fraction': fraction,
                            'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'source': str(source.resolve()), 'supervision_split': 'train', 'count': len(records),
            'validation_loaded_for_loss': False, 'test_loaded_for_loss': False, 'frames': records}
