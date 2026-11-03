"""Per-view atomic, hash-bound inference saves; preserve incomplete/failed files."""
import os
from pathlib import Path
import json
import numpy as np
from ..prepare_data import read, sha


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def save_view(path, arrays, bindings, counts):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists() and not path.with_suffix('.json').exists(), 'Never overwrite saved inference'
    temporary = path.with_suffix('.tmp.npz')
    assert not temporary.exists(), 'Preserve interrupted temporary view for inspection'
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    receipt = {'bindings': bindings, 'output_sha256': sha(path), 'counts': counts}
    atomic_json(path.with_suffix('.json'), receipt)
    return receipt


def read_saved(path, uv, old, bindings):
    path = Path(path)
    receipt = read(path.with_suffix('.json'))
    assert receipt['bindings'] == bindings and receipt['output_sha256'] == sha(path)
    with np.load(path) as data:
        assert np.array_equal(data['uv'], uv) and np.array_equal(data['original'], old)
        assert np.isfinite(data['updated']).all() and (data['updated'] > 0).all()
        assert np.array_equal(data['updated'][~data['accepted']], old[~data['accepted']])
        assert np.array_equal(data['accepted'], data['updated'] != old)
    return receipt
