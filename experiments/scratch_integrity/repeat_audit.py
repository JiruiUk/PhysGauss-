"""Retain the failed gate and measure more unchanged-path CUDA repeats."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys
import shutil
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.scratch_integrity.run import branch
from variants.scratch_integrity import prefix as common
from variants.metric_tuning.runtime import read_json, write_json, sha256


def main():
    torch.set_num_threads(4)
    report = ROOT / 'experiments/scratch_integrity'
    retained = report / 'preflight_attempt1'
    retained.mkdir(exist_ok=True)
    for name in ['training_freeze.json', 'training.log']:
        if not (retained / name).exists():
            shutil.copyfile(report / name, retained / name)
    for name in ['controller.json', 'pipeline.json']:
        if not (retained / name).exists():
            shutil.copyfile(common.RUNS / name, retained / name)
    binding = read_json(report / 'data_binding.json')
    original = ROOT / 'variants/color_repair/code'
    common.wire(original)
    deadline = datetime.now(timezone.utc) + timedelta(hours=1)
    names = ['flags_off', 'C1_original_path', 'C1_original_repeat']
    for name in ['C1_original_repeat2', 'C1_original_repeat3']:
        directory = branch(name, False, binding, True)
        if common.step(directory) < 15010:
            assert common.train(directory, original, 15010, deadline) == 75
        assert common.step(directory) == 15010
        names.append(name)
    states = {n: torch.load(common.RUNS / '_smoke' / n / 'rubber_duck/checkpoints/resume.pt', map_location='cpu', weights_only=False) for n in names}
    reference = states['C1_original_path']
    rows = {}
    for name, state in states.items():
        rows[name] = {}
        for group in ['gaussian_parameters', 'gaussian_buffers', 'velocity_net', 'grid4d', 'deform']:
            values = {}
            for key, value in state[group].items():
                delta = (value.double() - reference[group][key].double()).flatten()
                if delta.numel():
                    values[key] = {'different_elements': int((delta != 0).sum()), 'max_abs': float(delta.abs().max()), 'RMS': float(delta.square().mean().sqrt())}
            rows[name][group] = values
    write_json(report / 'unchanged_repeat_audit.json', {
        'reference': 'C1_original_path', 'all_iterations': 15010, 'differences': rows,
        'checkpoints_sha256': {n: sha256(common.RUNS / '_smoke' / n / 'rubber_duck/checkpoints/resume.pt') for n in names}})
    print('UNCHANGED REPEAT AUDIT SAVED', flush=True)


if __name__ == '__main__':
    main()
