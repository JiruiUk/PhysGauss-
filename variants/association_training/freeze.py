"""Immutable training provenance, checked before and after the matched pilot."""
from pathlib import Path
from variants.metric_tuning.runtime import read_json, write_json, sha256

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / 'experiments/association_training/training_freeze.json'
CORE = ['model.py', 'runtime.py', 'prepare.py', 'prepare_data.py', 'probe.py']


def core_hashes():
    return {name: sha256(Path(__file__).parent / name) for name in CORE}


def freeze():
    if LOCK.exists():
        record = read_json(LOCK)
        for name, digest in record['files'].items():
            assert sha256(ROOT / name) == digest, 'Frozen training input changed: ' + name
        return record
    files = read_json(ROOT / 'experiments/object_membership/input_lock.json')
    for name, digest in files.items():
        assert sha256(ROOT / name) == digest, 'Prior association input changed: ' + name
    flow_root = ROOT / 'data/derived/flow_comparison_train_only'
    manifest = read_json(flow_root / 'manifest.json')
    for name, digest in manifest['output_sha256'].items():
        path = flow_root / name
        assert sha256(path) == digest
        files[str(path.relative_to(ROOT))] = digest
    paths = list((ROOT / 'variants').rglob('*.py'))
    paths = [p for p in paths if '__pycache__' not in p.parts and
             not (p.parent == Path(__file__).parent and p.name in ['evaluate.py', 'finalize.py', 'verify.py'])]
    paths += [ROOT / p for p in ['tests/test_association_training.py',
              'experiments/association_training/plan.json',
              'runs/pidg_association_training/training_data.npz',
              'output/object_membership/membership.npz',
              'output/surface_correspondence/fragment_tracks.npz']]
    paths += list((ROOT / 'data/derived/kinematic_resplit/training').rglob('*.png'))
    paths += list((ROOT / 'data/derived/kinematic_resplit/training').glob('transforms*.json'))
    for path in paths:
        files[str(path.relative_to(ROOT))] = sha256(path)
    record = {'files': files, 'core_sha256': core_hashes(),
              'note': 'Created after successful preflight, before formal pilot. Training core unchanged since preflight; evaluation scripts excluded for independent development.'}
    write_json(LOCK, record)
    return record
