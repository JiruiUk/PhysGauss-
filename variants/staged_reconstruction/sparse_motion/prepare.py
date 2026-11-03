"""Freeze a separate sparse-motion representation experiment from S3-U final6k."""
import argparse
import numpy as np
from ..prepare_data import ROOT, REPORT, RUNS, read, write, sha
from ..motion_integrity.prepare import PARENT, PARENT_SHA
from .core import controls_from_canonical

EXP = REPORT/'sparse_motion'
WORK = RUNS/'sparse_motion'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--enable-sparse-motion', action='store_true')
    args = p.parse_args()
    if not args.enable_sparse_motion:
        raise SystemExit('Sparse motion is opt-in; pass --enable-sparse-motion')
    assert not (EXP/'protocol.json').exists(), 'Preserve earlier protocols'
    import torch
    assert sha(PARENT) == PARENT_SHA
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    chosen, xyz, radius = controls_from_canonical(parent['model']['pools.0._xyz'].numpy(), 256)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK/'controls.npz', rows=chosen, xyz=xyz, radius=radius)
    prior = read(REPORT/'motion_integrity/protocol.json')
    plan = {**prior, 'arm': 'I2_sparse_motion', 'controls': (WORK/'controls.npz').relative_to(ROOT).as_posix(),
            'controls_sha256': sha(WORK/'controls.npz'), 'control_count': len(xyz), 'control_knn': 4,
            'control_coordinates_and_radius_frozen': True, 'motion_mix_ramp_updates': 400,
            'actual_updates': 2000, 'comparison': 'Same final6k parent, full optimizer/RNG, same2k budget and I1 prior; different motion representation',
            'no_new_trainable_parameters': True, 'shared_full_center_mapping_for_render_and_physics': True}
    write(EXP/'protocol.json', plan)
    print('SPARSE_MOTION_PREPARED', len(xyz), flush=True)


if __name__ == '__main__':
    main()
