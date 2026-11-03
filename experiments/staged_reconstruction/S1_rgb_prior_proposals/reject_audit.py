"""Descriptive rejection accounting, no GT, threshold changes or retraining."""
from pathlib import Path
from collections import Counter
import numpy as np
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.rgb_prior_proposals.prepare import REPORT, RUNS, verify


def main():
    verify(); receipt = read(REPORT/'inference.json'); results = {}
    for case in ['diffuse', 'glossy']:
        counts = Counter()
        for i in range(96):
            path = RUNS/case/f'view_{i:03d}.npz'
            with np.load(path) as d:
                eligible, proposal = d['eligible'], d['proposed']; finite = eligible & np.isfinite(proposal)
                fit = finite & np.isfinite(d['fit_new']) & np.isfinite(d['fit_old']) & (d['fit_new'] <= d['fit_old']+1e-6)
                held_finite = np.isfinite(d['held_new']).all(0) & np.isfinite(d['held_old']).all(0)
                held_nonworse = held_finite & (d['held_new'] <= d['held_old']+1e-6).all(0)
                held_improve = held_finite & (d['held_new'] < d['held_old']-1e-6).any(0)
                counts.update({'eligible': int(eligible.sum()), 'finite_proposal': int(finite.sum()),
                    'fit_nonworse': int(fit.sum()), 'held_nonworse_among_finite': int((finite&held_nonworse).sum()),
                    'fit_and_held_nonworse': int((fit&held_nonworse).sum()),
                    'fit_held_strict': int((fit&held_nonworse&held_improve).sum()), 'accepted': int(d['accepted'].sum())})
        results[case] = dict(counts)
    for path, h in receipt['outputs'].items(): assert sha(ROOT/path) == h, path
    write(REPORT/'rejection_accounting.json', {'GT_read': False, 'diagnostic_only_not_geometry_correctness': True,
        'results': results, 'no_gate_or_threshold_changed': True, 'source_sha256': sha(Path(__file__))})
    print('RGB_PROPOSAL_REJECTION_ACCOUNTING', results, flush=True)


if __name__ == '__main__': main()
