"""Summarize actual native train support, separately from model completeness."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.uncertainty_motion.selection import compare_seeds


def main():
    exp = ROOT/'experiments/staged_reconstruction/uncertainty_motion'
    out = ROOT/'output/staged_reconstruction/uncertainty_motion'
    assessment = read(exp/'anchor_selection/assessment.json')
    verification = read(exp/'verification_native120.json')
    assert verification['passed']
    a = np.load(out/'anchor_selection/anchors.npz')
    assert sha(out/'anchor_selection/anchors.npz') == assessment['anchor_arrays_sha256']
    frames = a['frames']; stable = a['stable_anchors']; per_seed = a['per_seed_selections']
    trusted_counts = a['trusted'][:, stable].sum(1)
    overlap = compare_seeds(per_seed[0], per_seed[1])
    period = a['per_seed_period'].min(0)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    axes[0].plot(frames, per_seed[0].sum(1), label='Qualified top 2%, seed A', color='#5f7996')
    axes[0].plot(frames, trusted_counts, label='Trusted recurring anchors', color='#aa5d1b')
    axes[0].set(xlabel='Actual training frame', ylabel='Gaussian count', title='Observation support over time')
    axes[0].legend(fontsize=8)
    axes[1].hist(period[period > 0], bins=np.arange(.5, 121.5, 3), color='#5f7996')
    axes[1].axvline(5, color='#aa5d1b', ls='--', label='Minimum significant period = 5')
    axes[1].set(xlabel='Minimum of the two seed significant periods', ylabel='Gaussian count', title='Recurrence before voxel selection')
    axes[1].legend(fontsize=7)
    axes[2].plot(frames, overlap, color='#547a54')
    axes[2].axhline(.8, color='#aa5d1b', ls='--', label='Predeclared feasibility gate')
    axes[2].set(xlabel='Actual training frame', ylabel='Jaccard of independently qualified sets', ylim=(0, 1.02), title='Independent selection agreement')
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('S3-U native1600 train-only audit: color support is not geometry or physical identity', fontsize=11)
    figure = out/'anchor_selection/reliability_summary_v2.png'
    assert not figure.exists()
    fig.savefig(figure, dpi=170); plt.close(fig)
    native_stats = next(v for v in verification['observed_audits'] if v['name'] == 'observed_native120')
    outcome = 'PASS for graph feasibility only' if assessment['passed_for_graph_feasibility_only'] else 'FAIL graph feasibility'
    gates = '\n'.join(f'|{name}|{passed}|' for name, passed in assessment['gates'].items())
    text = f'''# Native train reliability and recurring anchors: {outcome}

The exact user-selected S3-U final6k parent was audited at all120 observed
training times using the unchanged native1600 renderer. Both independent
1536-probe estimates per frame were checked against192 population-spanning
one-hot sample rows. Every frame passed the registered measurement gates.
Independent verification recomputes exact-support errors from saved arrays;
the median across frame/seed medians is{native_stats['exact_support_error_median_across_cases_and_seeds']:.4%},
and the worst frame/seed median is{native_stats['worst_case_median_relative_error']:.4%}.
Original native RGB/alpha are exact; all model tensors and global RNG remain
unchanged. Actual optimizer updates:0. Test RGB read:False.

## Registered selection results

Spatial candidates use voxels twice the original initialization spacing.
Qualified top2% ranks are computed separately for both seeds, then candidates
with at least5 significant frames are intersected. No absent observation is
filled to meet a count. Physical body count is not inferred.

- Seed anchor counts:{assessment['per_seed_anchor_counts']}.
- Stable intersection:{assessment['stable_anchor_count']} /15,507 dynamic rows.
- Minimum currently trusted stable anchors in an observed frame:{assessment['trusted_anchors_each_frame_min']}.
- Anchors recurring at least5 times within early/middle/late thirds:{assessment['recurring_anchors_by_temporal_third']}.
- Independent per-frame selection Jaccard: minimum{assessment['independent_seed_selection_jaccard_min']:.4f},
  median{assessment['independent_seed_selection_jaccard_median']:.4f}.
- Fraction of qualified model points within the registered radius of a stable
  anchor: minimum{assessment['model_support_neighbor_coverage_min']:.4%},
  median{assessment['model_support_neighbor_coverage_median']:.4%}.
  This describes S3-U support and is not GT surface coverage.

|Registered gate|Passed|
|---|---|
{gates}

## Interpretation and next action

This establishes which color-supported point rows can be tested as motion
anchors. It does not prove their depth or material identity. Graph feasibility
is distinct from training preflight and the complete-object goal. No graph loss
or new model has been trained in this milestone. The full6x9 then6x200 novel-view
acceptance remains unfulfilled; no model is promoted.

Any following motion intervention must retain uncertain depth and avoid claiming
that nearby rows belong to one object. It needs separate opt-in tests, native
flags-off parity, actual short training/recovery checks, matched continuation
controls and actual novel-view image review. I10's numerical repeat sensitivity
still prevents interpreting a single PSNR difference as a stable causal effect.

Evidence: `anchor_selection/assessment.json`, `observed_native120/result.json`,
`verification_native120.json`, `observed_native120_protocol.json`,
`output/staged_reconstruction/uncertainty_motion/anchor_selection/anchors.npz`.
Source audit, adaptations and failed low-sample/wrapper attempts are retained
in `research.md`, `measurement_assessment.md` and versioned protocol/log files.
'''
    path = exp/'assessment_v2.md'; assert not path.exists(); path.write_text(text, encoding='utf-8')
    write(exp/'report_receipt_v2.json', {'script_sha256': sha(__file__), 'assessment_sha256': sha(path),
        'figure_sha256': sha(figure), 'selection_assessment_sha256': sha(exp/'anchor_selection/assessment.json'),
        'native_verification_sha256': sha(exp/'verification_native120.json'), 'optimizer_updates': 0})
    print('NATIVE_RELIABILITY_REPORT_SAVED', outcome, flush=True)


if __name__ == '__main__':
    main()
