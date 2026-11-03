"""Independent saved-output/input boundary replay, without evaluator labels."""
from pathlib import Path
from collections import Counter
import numpy as np
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.pretrained_depth.prepare import BASE, REPORT, RUNS, PARENT, SOURCE, verify
from variants.staged_reconstruction.pretrained_depth.core import calibrate, split_anchors


def main():
    verify(); p = read(BASE / 'protocol.json'); receipt = read(REPORT / 'inference.json')
    results = {}
    for case in p['synthetic_cases']:
        near, far = read(SOURCE / f'{case}_inference.json')['camera_derived_depth_range']
        reasons = Counter(); counts = Counter()
        for i in range(96):
            path = RUNS / case / f'view_{i:03d}.npz'
            saved_info = read(path.with_suffix('.json'))
            assert saved_info['output_sha256'] == sha(path)
            with np.load(path) as saved, np.load(PARENT / case / f'view_{i:03d}.npz') as anchors:
                raw, depth = saved['raw_relative'], saved['aligned_depth']
                assert raw.shape == (144, 256) and np.isfinite(raw).all() and (raw >= 0).all()
                tight = anchors['tight']; uv = anchors['uv'][tight]; z = anchors['depth'][tight]
                assert np.array_equal(uv, saved['anchor_uv']) and np.array_equal(z, saved['anchor_depth'])
                fresh, info = calibrate(None, raw, uv, z, near, far, p, True)
                assert np.array_equal(fresh, depth)
                for key in ['fit_ids', 'held_ids']: assert np.array_equal(info[key], saved[key])
                for key, value in info.items():
                    if key not in ['fit_ids', 'held_ids']: assert value == saved_info[key], key
                assert not np.intersect1d(saved['fit_ids'], saved['held_ids']).size
                assert not split_anchors(uv[saved['fit_ids']]).any()
                assert split_anchors(uv[saved['held_ids']]).all()
                valid = depth > 0
                assert ((depth[valid] >= near) & (depth[valid] <= far)).all()
                if not info['calibrated']: assert not valid.any()
                reasons[info['reason']] += 1; counts['views'] += 1
                counts['anchors'] += len(uv); counts['valid_pixels'] += int(valid.sum())
        results[case] = {'reasons': dict(reasons), 'counts': dict(counts)}
    for name, h in receipt['outputs'].items(): assert sha(ROOT / name) == h, name
    write(REPORT / 'input_audit.json', {'passed': True, 'GT_read': False, 'all192_views_replayed_exact': True,
          'unchanged_D2_tight_anchors_and_disjoint_fit_check': True, 'results': results,
          'independent_audit_source_sha256': sha(Path(__file__)), 'inference_sha256': sha(REPORT / 'inference.json')})
    print('PRETRAINED_INPUT_AUDIT_PASS', results, flush=True)


if __name__ == '__main__':
    main()
