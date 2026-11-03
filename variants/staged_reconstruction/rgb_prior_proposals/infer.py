"""One opt-in RGB proposal run, retaining exact full candidate membership."""
import argparse
import os
import msvcrt
import numpy as np
import torch
from ..prepare_data import ROOT, read, write, sha
from ..synthetic_audit.input import load
from ..adaptive_static.geometry import patch_features
from ..pretrained_depth.model import configure
from .prepare import BASE, REPORT, RUNS, D1, D2, D3, SOURCE, verify, stopped
from .compute import compute_view
from .io import save_view, read_saved


def inputs(case, i, train, row, sources):
    return {'case': case, 'index': i, 'sources': sources, 'RGB_sha256': sha(train/row['image']),
        'D1_sha256': sha(D1/case/f'view_{i:03d}.npz'), 'D2_sha256': sha(D2/case/f'view_{i:03d}.npz'),
        'Small_container_sha256': sha(D3/case/f'view_{i:03d}.npz'), 'source_freeze_sha256': sha(REPORT/'source_freeze.json')}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-rgb-prior-proposals', action='store_true'); args = parser.parse_args()
    assert args.enable_rgb_prior_proposals and not stopped()
    verify(); assert not (REPORT/'inference.json').exists(), 'Preserve completed candidate'
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS/'inference.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    configure(); results = {}
    try:
        protocol = read(BASE/'protocol.json')
        for case in protocol['synthetic_cases']:
            train = ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train'
            values = load(train); meta = read(train/'cameras.json')
            source = read(SOURCE/f'{case}_inference.json'); near, far = source['camera_derived_depth_range']
            images = torch.from_numpy(values['rgb']).permute(0, 3, 1, 2).cuda()
            features, texture = patch_features(images, 5); del images
            views = torch.as_tensor(values['views'], dtype=torch.float32, device='cuda')
            k = torch.as_tensor(values['K'], dtype=torch.float32, device='cuda')
            directory = RUNS/case; directory.mkdir(exist_ok=True); rows = []
            for i, row in enumerate(meta['frames']):
                assert not stopped(), 'Saved user stop; retain all views and release GPU'
                path = directory/f'view_{i:03d}.npz'; bindings = inputs(case, i, train, row, source['sources'][i])
                with np.load(D1/case/f'view_{i:03d}.npz') as saved: depth = saved['refined'].copy()
                yy, xx = np.nonzero(depth > 0); uv = np.c_[xx, yy]; old = depth[yy, xx]
                if path.exists():
                    receipt = read_saved(path, uv, old, bindings)
                else:
                    # Only the raw relative array is read; aligned_depth is not an input.
                    with np.load(D3/case/f'view_{i:03d}.npz') as saved: raw = saved['raw_relative'].copy()
                    with np.load(D2/case/f'view_{i:03d}.npz') as saved:
                        anchors = {key: saved[key].copy() for key in ['uv', 'depth', 'tight', 'lower', 'upper']}
                    arrays, counts = compute_view(depth, anchors, raw, features, texture, views, k, i,
                        source['sources'][i], near, far, protocol)
                    receipt = save_view(path, arrays, bindings, counts)
                rows.append(receipt['counts'])
                write(RUNS/'state.json', {'status': 'inference', 'pid': os.getpid(), 'case': case, 'completed': i+1,
                    'total': 96, 'accepted_so_far': sum(r['accepted'] for r in rows)})
                if i % 12 == 0: print('RGB_PRIOR_PROPOSALS', case, i+1, 'accepted', sum(r['accepted'] for r in rows), flush=True)
            results[case] = {'views': len(rows), 'totals': {key: sum(r[key] for r in rows) for key in rows[0]}, 'per_view': rows}
            del features, texture, views, k; torch.cuda.empty_cache()
        verify(); write(REPORT/'inference.json', {'GT_read': False, 'Gaussian_training': False,
            'results': results, 'source_manifest_sha256': sha(REPORT/'source_freeze.json'),
            'all_D1_candidates_preserved': True, 'no_aligned_or_oracle_depth_used': True,
            'outputs': {p.relative_to(ROOT).as_posix(): sha(p) for p in RUNS.rglob('view_*') if p.is_file()}})
        write(RUNS/'state.json', {'status': 'inferred_pending_independent_evaluation'})
        print('RGB_PRIOR_PROPOSALS_COMPLETE', {case: r['totals'] for case, r in results.items()}, flush=True)
    except Exception as error:
        write(RUNS/'state.json', {'status': 'saved_stopped' if stopped() else 'failed', 'error': str(error), 'pid': os.getpid()})
        raise
    finally:
        torch.cuda.empty_cache()
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__': main()
