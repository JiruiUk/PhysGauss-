"""Prepare an immutable training-only ordinal prior and matched protocol."""
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from ..prepare_data import ROOT, REPORT, RUNS, DATA, read, write, sha
from ..stage2.train import image
from ..union.core import dynamic_mask
from .core import pair_bank

EXP = REPORT/'ordinal_depth'
WORK = RUNS/'ordinal_depth'


def main():
    assert not (EXP/'protocol.json').exists(), 'Preserve protocol versions'
    audit = read(EXP/'data_audit.json')
    assert audit['parent_unchanged'] and audit['training_only']
    assert all(r['native_depth_gradient']['finite'] and r['native_depth_gradient']['canonical_norm'] > 0 for r in audit['rows'])
    assert all(r['static_pseudo_vs_normalized_render_spearman'] < -.3 for r in audit['rows'])
    frames = read(DATA/'transforms_train.json')['frames']
    assert len(frames) == 120
    (WORK/'pairs').mkdir(parents=True, exist_ok=True)
    receipts = {}
    for frame in frames:
        number = int(Path(frame['file_path']).stem)
        path = ROOT/f'data/PIDG/rubber_duck/depth_distill/{number:04d}.npy'
        pseudo = cv2.resize(np.load(path), (800, 450), interpolation=cv2.INTER_AREA)
        rgba = image(frame, 800)
        mask = binary_erosion(dynamic_mask(rgba), iterations=2)
        pairs, signs = pair_bank(pseudo, mask, 73000+number)
        out = WORK/f'pairs/{number:04d}.npz'
        np.savez_compressed(out, pairs=pairs, signs=signs)
        receipts[str(number)] = {'depth_sha256': sha(path), 'pairs_sha256': sha(out), 'pairs': len(pairs), 'foreground_pixels': int(mask.sum())}
    plan = read(REPORT/'motion_integrity/protocol.json')
    plan.update(arm='I3_ordinal_depth', depth_width=800, depth_pairs_per_step=4096,
                ordinal_weight=.03, ordinal_ramp=100, ordinal_sharpness=10.,
                pseudo_depth_direction=-1, smoke_updates=200,
                additional_prior='Cached Distill Any Depth inverse-depth ordering, training120 only',
                matched_control='I1_curve_isometry final2k; same parent, optimization state, cameras, image and physics losses',
                geometry_only_depth_gradients=True, depth_receipts=receipts)
    write(EXP/'protocol.json', plan)
    print('ORDINAL_PREPARED', len(receipts), flush=True)


if __name__ == '__main__':
    main()
