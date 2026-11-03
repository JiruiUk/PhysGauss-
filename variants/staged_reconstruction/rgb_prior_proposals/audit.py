"""All-view saved-state audit plus independent native RGB proposal replay, no GT."""
from pathlib import Path
import numpy as np
import torch
from ..prepare_data import ROOT, read, write, sha
from ..synthetic_audit.input import load
from ..adaptive_static.geometry import patch_features
from ..pretrained_depth.model import configure
from .prepare import BASE, REPORT, RUNS, D1, D2, D3, SOURCE, verify, stopped
from .core import ordinal_bounds, eligible_old_depth, accept_rgb
from .compute import compute_view
from .infer import inputs
from .io import read_saved


def main():
    assert not stopped(); verify(); receipt = read(REPORT/'inference.json'); p = read(BASE/'protocol.json')
    for path, h in receipt['outputs'].items(): assert sha(ROOT/path) == h, path
    configure(); results = {}
    for case in p['synthetic_cases']:
        train = ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train'; meta = read(train/'cameras.json')
        source = read(SOURCE/f'{case}_inference.json'); near, far = source['camera_derived_depth_range']
        for i, row in enumerate(meta['frames']):
            assert not stopped()
            path = RUNS/case/f'view_{i:03d}.npz'
            with np.load(D1/case/f'view_{i:03d}.npz') as d1: depth = d1['refined']
            yy, xx = np.nonzero(depth > 0); uv = np.c_[xx, yy]; old = depth[yy, xx]
            read_saved(path, uv, old, inputs(case, i, train, row, source['sources'][i]))
            with np.load(D2/case/f'view_{i:03d}.npz') as a, np.load(D3/case/f'view_{i:03d}.npz') as prior, np.load(path) as saved:
                tight = a['tight']; lo, hi, known, lb, hb = ordinal_bounds(uv, prior['raw_relative'], a['uv'][tight],
                    a['lower'][tight], a['upper'][tight], near, far, p)
                for name, value in [('lower', lo), ('upper', hi), ('known', known), ('lower_anchor_blocks', lb), ('upper_anchor_blocks', hb)]:
                    assert np.array_equal(saved[name], value), name
                eligible = eligible_old_depth(old, lo.astype(old.dtype), hi.astype(old.dtype), known)
                assert np.array_equal(saved['eligible'], eligible)
                # Replay the acceptance equation independently from saved RGB costs.
                valid = torch.as_tensor(np.isfinite(saved['proposed']))
                out, mask = accept_rgb(torch.from_numpy(old), torch.as_tensor(saved['proposed']),
                    torch.as_tensor(saved['fit_old']), torch.as_tensor(saved['fit_new']),
                    torch.as_tensor(saved['held_old']), torch.as_tensor(saved['held_new']), valid, True)
                assert np.array_equal(mask.numpy(), saved['accepted']) and np.array_equal(out.numpy(), saved['updated'])
        # Replay one actual view in this independently launched process. Prefer
        # a view with an accepted change, then any eligible view; no GT involved.
        rows = receipt['results'][case]['per_view']
        chosen = next((i for i, r in enumerate(rows) if r['accepted']), next((i for i, r in enumerate(rows) if r['eligible']), 0))
        values = load(train)
        images = torch.from_numpy(values['rgb']).permute(0, 3, 1, 2).cuda()
        features, texture = patch_features(images, 5); del images
        views = torch.as_tensor(values['views'], dtype=torch.float32, device='cuda')
        k = torch.as_tensor(values['K'], dtype=torch.float32, device='cuda')
        with np.load(D1/case/f'view_{chosen:03d}.npz') as d: depth = d['refined'].copy()
        with np.load(D2/case/f'view_{chosen:03d}.npz') as d: anchors = {key: d[key].copy() for key in ['uv', 'depth', 'tight', 'lower', 'upper']}
        with np.load(D3/case/f'view_{chosen:03d}.npz') as d: raw = d['raw_relative'].copy()
        fresh, counts = compute_view(depth, anchors, raw, features, texture, views, k, chosen, source['sources'][chosen], near, far, p)
        with np.load(RUNS/case/f'view_{chosen:03d}.npz') as saved:
            for name, value in fresh.items(): assert np.array_equal(value, saved[name], equal_nan=True), (case, name)
        results[case] = {'all96_views_input_bounds_acceptance_audited': True, 'replayed_native_view': chosen,
            'native_RGB_all_outputs_and_costs_exact': True, 'replayed_counts': counts}
        del features, texture, views, k; torch.cuda.empty_cache()
    verify(); write(REPORT/'input_audit.json', {'passed': True, 'GT_read': False, 'all_D1_candidates_and_unaccepted_depths_exact': True,
        'independent_process_actual_native_RGB_replay': True, 'results': results, 'source_sha256': sha(Path(__file__)),
        'inference_sha256': sha(REPORT/'inference.json')})
    print('RGB_PROPOSAL_INPUT_AUDIT_PASS', results, flush=True)


if __name__ == '__main__': main()
