"""Train-only provenance, atomic JSON, explicit stop checks and file locks."""
from contextlib import contextmanager
import json
import msvcrt
import os
import sys
from pathlib import Path
from ..prepare_data import ROOT, read, sha

BASE = ROOT/'experiments/staged_reconstruction/S1_joint_rgb'
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_joint_rgb/v1'
OUT = ROOT/'output/staged_reconstruction/S1_joint_rgb/v1'


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    os.replace(temp, path)


def stopped():
    return any(p.exists() for p in [RUNS/'stop.request', RUNS.parent/'stop.request',
        RUNS.parent.parent/'stop.request', ROOT/'stop.request'])


@contextmanager
def lock(name):
    RUNS.mkdir(parents=True, exist_ok=True)
    stream = (RUNS/name).open('a+b'); stream.seek(0)
    if not stream.read(1):
        stream.write(b'0'); stream.flush()
    stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        yield
    finally:
        stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1); stream.close()


def verify():
    for name, expected in read(BASE/'preregistration.json')['files'].items():
        assert sha(ROOT/name) == expected, name
    for name, expected in read(REPORT/'source_freeze.json').items():
        assert sha(ROOT/name) == expected, name


def freeze():
    assert not stopped() and not (REPORT/'source_freeze.json').exists()
    protocol = read(BASE/'protocol.json'); files = {}
    for case in protocol['cases']:
        parent = protocol['parents'][case]
        assert sha(ROOT/parent['checkpoint']) == parent['sha256']
        paths = [ROOT/parent['checkpoint'], ROOT/parent['initialization']]
        paths += list((ROOT/protocol['synthetic_data']/case/'train').rglob('*'))
        for path in paths:
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = sha(path)
    # The J0 native import receipt binds the unchanged renderer, original grids,
    # networks, CUDA adapter and Gaussian helpers used by this experiment.
    inherited = read(ROOT/'experiments/staged_reconstruction/S1_joint_revocation_audit/v1/preflight.json')['imported_sources']
    for name, expected in inherited.items():
        assert sha(ROOT/name) == expected, name
        files[name] = expected
    for module in list(sys.modules.values()):
        source = getattr(module, '__file__', None)
        if not source:
            continue
        path = Path(source).resolve()
        if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'envs'):
            files[path.relative_to(ROOT).as_posix()] = sha(path)
    for directory in [ROOT/'variants/staged_reconstruction/joint_rgb', BASE]:
        for path in directory.glob('*.py'):
            files[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in [BASE/'plan.md', BASE/'protocol.json', BASE/'preregistration.json',
        ROOT/'tests/test_joint_rgb.py', ROOT/'variants/staged_reconstruction/nuisance_static/runtime.py',
        ROOT/'variants/staged_reconstruction/synthetic_audit/input.py',
        ROOT/'variants/staged_reconstruction/adaptive_static/model.py']:
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    assert not any('/evaluation_only/' in name for name in files)
    atomic_json(REPORT/'source_freeze.json', files)
    print('J1_TRAIN_ONLY_FREEZE', len(files), sha(REPORT/'source_freeze.json'), flush=True)
