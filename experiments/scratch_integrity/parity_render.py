"""Measure native rendered outputs of the disabled path and original repeats."""
from pathlib import Path
import os
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'variants/association_training/code'))
os.environ['PIDG_HASH_SOURCE_DIR'] = str(ROOT / 'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR'] = str(ROOT / 'third_party/PIDG/tmp_build')
from variants.association_training.evaluate import load, render_call
from variants.shape_coherence.observations import camera
from variants.flow_comparison.temporal import warp_at
from variants.metric_tuning.runtime import read_json, write_json, sha256


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    report = ROOT / 'experiments/scratch_integrity'
    runs = ROOT / 'runs/pidg_scratch_integrity/_smoke'
    meta = read_json(ROOT / 'data/derived/kinematic_resplit/training/transforms_train.json')
    frames = sorted(meta['frames'], key=lambda f: int(Path(f['file_path']).stem))
    chosen = [frames[i] for i in np.linspace(0, len(frames) - 1, 10).astype(int)]
    names = ['C1_original_path', 'flags_off', 'C1_original_repeat', 'C1_original_repeat2', 'C1_original_repeat3']
    reference = []
    results = {}
    proofs = {}
    for name in names:
        directory = runs / name / 'rubber_duck'
        os.environ['PIDG_METRIC_CONFIG'] = str(directory / 'metric_config.json')
        from variants.color_repair.color import _read
        _read.cache_clear()
        g, d, proof = load(directory)
        proofs[name] = proof
        rows = []
        for index, frame in enumerate(chosen):
            n = int(Path(frame['file_path']).stem)
            package = render_call(camera(frame, meta, 1600), g, warp_at(d, g.get_xyz.detach(), (n - 1) / 199))
            arrays = {k: package[k].detach().cpu().numpy() for k in ['render', 'alpha']}
            if name == 'C1_original_path':
                reference.append(arrays)
            row = {'frame': n}
            for key in arrays:
                delta = arrays[key].astype(float) - reference[index][key].astype(float)
                row[key] = {'max_abs': float(np.abs(delta).max()), 'RMS': float(np.sqrt(np.mean(delta ** 2)))}
            rows.append(row)
        results[name] = {'frames': rows, 'aggregate_RMS': {k: float(np.sqrt(np.mean([r[k]['RMS'] ** 2 for r in rows]))) for k in ['render', 'alpha']}}
        del g, d, package
        torch.cuda.empty_cache()
    write_json(report / 'flags_off_native_render_audit.json', {
        'native_resolution': [1600, 900], 'training_views_only': True,
        'states': proofs, 'results': results, 'script_sha256': sha256(Path(__file__))})
    print('NATIVE PARITY AUDIT SAVED', flush=True)


if __name__ == '__main__':
    main()
