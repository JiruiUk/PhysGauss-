"""Independent-process per-view inference recovery, no formal dataset or GT."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from ..prepare_data import ROOT, write, read, sha
from ..adaptive_static.geometry import patch_features
from .core import propose
from .io import save_view, read_saved

BASE = ROOT/'experiments/staged_reconstruction/S1_rgb_prior_proposals'
WORK = ROOT/'runs/pidg_staged_reconstruction/S1_rgb_prior_proposals/preflight'


def worker(directory, limit):
    torch.set_num_threads(4)
    width = 64
    views = torch.eye(4).repeat(7, 1, 1)
    views[:, 0, 3] = torch.tensor([0., 8/35, 16/35, 24/35, -8/35, -16/35, -24/35])
    k = torch.tensor([[35., 0, 31.5], [0, 35., 31.5], [0, 0, 1.]])
    yy, xx = torch.meshgrid(torch.arange(width), torch.arange(width), indexing='ij')
    images = []
    for v in views:
        wx = (xx-31.5)/35*4-v[0, 3]; wy = (yy-31.5)/35*4
        gray = .45+.15*torch.sin(wx*7)+.1*torch.cos(wy*9+wx*3)
        images.append(torch.stack([gray, gray*.9, gray*.8]))
    features, texture = patch_features(torch.stack(images), 5)
    uv = torch.tensor([[32., 30.], [29., 35.]])
    directory.mkdir(parents=True, exist_ok=True); processed, restored = 0, 0
    for i in range(limit):
        old = torch.tensor([4.8+i*.01, 4.8+i*.01])
        path = directory/f'view_{i:03d}.npz'
        bindings = {'fixture_index': i, 'core_sha256': sha(Path(__file__).with_name('core.py'))}
        if path.exists():
            read_saved(path, uv.numpy(), old.numpy(), bindings); restored += 1
            continue
        same, off = propose(None, old)
        assert same is old and not off['accepted'].any()
        result, d = propose(uv, old, torch.tensor([3.8, 3.8]), torch.tensor([4.2, 4.2]),
            torch.ones(2, dtype=torch.bool), features, texture, views, k, 0, list(range(1, 7)), 2, 6, True)
        assert d['accepted'].all() and (result-4).abs().max() < .03
        arrays = {'uv': uv.numpy(), 'original': old.numpy(), 'updated': result.numpy(),
                  **{key: value.numpy() for key, value in d.items()}}
        save_view(path, arrays, bindings, {'accepted': int(d['accepted'].sum())}); processed += 1
    write(directory/f'worker_{os.getpid()}.json', {'pid': os.getpid(), 'processed': processed, 'restored': restored, 'limit': limit})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--worker'); parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--run-preflight', action='store_true'); args = parser.parse_args()
    if args.worker: return worker(Path(args.worker), args.limit)
    assert args.run_preflight and not (BASE/'v1/preflight.json').exists()
    WORK.mkdir(parents=True, exist_ok=True)
    pids = []
    for label, directory, limit in [('full', WORK/'full', 4), ('partial', WORK/'resumed', 2), ('resume', WORK/'resumed', 4)]:
        log = BASE/'v1'/f'preflight_{label}.log'
        with log.open('wb') as output:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'variants.staged_reconstruction.rgb_prior_proposals.preflight',
                '--worker', str(directory), '--limit', str(limit)], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            pids.append(child.pid); assert child.wait() == 0, log
    for i in range(4):
        with np.load(WORK/'full'/f'view_{i:03d}.npz') as full, np.load(WORK/'resumed'/f'view_{i:03d}.npz') as resumed:
            assert full.files == resumed.files
            for key in full.files: assert np.array_equal(full[key], resumed[key], equal_nan=True), key
    workers = [read(p) for p in (WORK/'resumed').glob('worker_*.json')]
    assert sorted((r['processed'], r['restored']) for r in workers) == [(2, 0), (2, 2)]
    off_cuda = False
    if torch.cuda.is_available():
        old = torch.tensor([4.], device='cuda'); same, d = propose(None, old)
        assert same is old and not d['accepted'].any(); del old, same, d
        torch.cuda.empty_cache(); off_cuda = True
    write(BASE/'v1/preflight.json', {'passed': True, 'independent_processes': pids,
        'native_RGB_plane_proposal_and_all_costs_replayed_exact': True, 'restored_views': 2,
        'CPU_and_CUDA_flags_off_identity': off_cuda, 'Gaussian_training_Adam_RNG_claim': False,
        'source_sha256': sha(Path(__file__)),
        'logs': {p.relative_to(ROOT).as_posix(): sha(p) for p in (BASE/'v1').glob('preflight_*.log')}})
    print('RGB_PROPOSAL_PREFLIGHT_PASS', pids, flush=True)


if __name__ == '__main__': main()
