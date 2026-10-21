"""Original full checkpoint plus a checksum-bound four-metric selection state."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('pidg_original_checkpoint', ROOT / 'reproduction/scripts/training_checkpoint.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def save_training_state(local, destination):
    manifest = base.save_training_state(local, destination)
    local['tuning'].save(destination, manifest)
    return manifest


def restore_training_state(path, gaussians, deform, opt, scene, tuning):
    state = base.restore_training_state(path, gaussians, deform, opt, scene)
    tuning.restore(path, state['iteration'])
    return state
