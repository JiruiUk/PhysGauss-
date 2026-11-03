"""Independent aggregate replay, evaluator-only coverage groups and controls."""
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha

BASE = ROOT/'experiments/staged_reconstruction/S1_plane_transport'
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_plane_transport/v1'
OUT = ROOT/'output/staged_reconstruction/S1_plane_transport/v1'
SOURCE = ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2'
D5 = ROOT/'experiments/staged_reconstruction/S1_photometric_oracle/v1'


def main():
    assert read(REPORT/'input_audit.json')['passed']
    assert not (REPORT/'exports.json').exists()
    relative = (SOURCE/'dataset.json').relative_to(ROOT).as_posix()
    assert sha(SOURCE/'dataset.json') == read(D5/'source_freeze.json')[relative]
    for name, h in read(SOURCE/'dataset.json')['files'].items():
        assert sha(ROOT/name) == h, name
    inference = read(REPORT/'inference.json')
    for name, h in inference['outputs'].items():
        assert sha(ROOT/name) == h, name
    evaluation = read(REPORT/'evaluation.json')
    diagnostics = {}
    for case, result in evaluation['results'].items():
        meta = read(ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train/cameras.json')
        counts = {arm: Counter() for arm in ['D6', 'D7']}
        errors = {arm: {'static': [], 'thin': []} for arm in counts}
        groups = {name: Counter() for name in ['all', 'parent_gross_or_no_static', 'visible_thin', 'without_target_slope']}
        rejection = Counter()
        for index, frame in enumerate(meta['frames']):
            with np.load(RUNS/case/f'view_{index:03d}.npz') as data:
                x, y = data['uv'].T
                old, new, accepted = data['original'], data['updated'], data['accepted']
                query = data['source_query_valid'].any(0)
                hypothesis = data['candidate_valid'].any(0)
                eligible = data['eligible']
                own = data['parent_slope_admitted']
                assert np.array_equal(old[~accepted], new[~accepted])
                with np.load(ROOT/f'data/derived/clean_static_synthetic/v2/{case}/evaluation_only/gt'/(Path(frame['image']).stem+'.npz')) as gt:
                    target = gt['static_depth'][y, x]
                    surface = target > 0
                    full_visible = gt['observed_static_visible'] & (gt['static_depth'] > 0)
                    full_thin = gt['thin_static_visible'] & full_visible
                    visible, thin = full_visible[y, x], full_thin[y, x]
                for arm, z in [('D6', old), ('D7', new)]:
                    error = abs(z-target)
                    counts[arm].update(candidates=len(z), on_static=int(surface.sum()),
                        gross_or_no_static=int((~surface | (error > .05)).sum()),
                        visible_total=int(full_visible.sum()), thin_total=int(full_thin.sum()),
                        accurate_visible_005=int((visible & (error <= .05)).sum()),
                        accurate_thin_002=int((thin & (error <= .02)).sum()))
                    errors[arm]['static'].append(error[surface])
                    errors[arm]['thin'].append(error[thin])
                for group, mask in [('all', np.ones(len(old), bool)),
                    ('parent_gross_or_no_static', ~surface | (abs(old-target) > .05)),
                    ('visible_thin', thin), ('without_target_slope', ~own)]:
                    groups[group].update(rows=int(mask.sum()), source_query=int((mask & query).sum()),
                        valid_hypothesis=int((mask & hypothesis).sum()), eligible=int((mask & eligible).sum()),
                        accepted=int((mask & accepted).sum()),
                        accepted_on_static=int((mask & accepted & surface).sum()),
                        improved=int((mask & accepted & surface & (abs(new-target) < abs(old-target))).sum()),
                        worsened=int((mask & accepted & surface & (abs(new-target) > abs(old-target))).sum()))
                fit_good = np.isfinite(data['fit_new']) & (data['fit_new'] <= data['fit_old']+1e-6)
                held_finite = np.isfinite(data['held_old']).all(0) & np.isfinite(data['held_new']).all(0)
                held_nonworse = (data['held_new'] <= data['held_old']+1e-6).all(0)
                held_strict = (data['held_new'] < data['held_old']-1e-6).any(0)
                rejection.update(eligible=int(eligible.sum()), accepted=int(accepted.sum()),
                    fit_worse_or_invalid=int((eligible & ~fit_good).sum()),
                    held_invalid=int((eligible & ~held_finite).sum()),
                    held_worse=int((eligible & held_finite & ~held_nonworse).sum()),
                    held_no_strict_improvement=int((eligible & held_finite & ~held_strict).sum()))
        for arm in counts:
            assert dict(counts[arm]) == result['arms'][arm]['counts']
            for key, saved in [('static', 'static_error'), ('thin', 'thin_error')]:
                values = np.concatenate(errors[arm][key])
                assert len(values) == result['arms'][arm][saved]['n']
                assert float(values.mean()) == result['arms'][arm][saved]['mean']
                assert float(np.percentile(values, 90)) == result['arms'][arm][saved]['p90']
        assert groups['all']['accepted'] == inference['results'][case]['totals']['accepted']
        assert groups['without_target_slope']['accepted'] == inference['results'][case]['totals']['accepted_without_target_slope']
        diagnostics[case] = {'groups': {k: dict(v) for k, v in groups.items()},
            'overlapping_rejection_causes_not_partition': dict(rejection)}
    controls = read(REPORT/'analytic_controls.json')['controls']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, (name, values) in zip(axes, controls.items()):
        x = np.arange(2)
        ax.scatter(x-.10, values['original_depth'], label='target before', color='#cb181d')
        ax.scatter(x, values['true_depth'], label='analytic truth', marker='x', color='#238b45', s=75)
        ax.scatter(x+.10, values['updated_depth'], label='D7 transported', color='#2171b5')
        ax.set_title(name); ax.set_xticks(x, ['pixel 1', 'pixel 2']); ax.set_ylabel('reference camera depth')
        ax.set_ylim(3.8, 4.95); ax.legend(fontsize=8)
    fig.suptitle('Fixed analytic controls: fixture source planes, NOT dataset accuracy')
    path = OUT/'analytic_transport_controls.png'
    assert not path.exists(); fig.savefig(path, dpi=150); plt.close(fig)
    write(REPORT/'coverage_diagnostic.json', {'results': diagnostics,
        'posthoc_evaluator_only_not_candidate_gate_or_training_input': True,
        'no_GT_pixel_arrays_exported_or_feedback': True, 'all_192_metric_count_MAE_P90_replay_exact': True,
        'evaluation_sha256': sha(REPORT/'evaluation.json'), 'source_sha256': sha(Path(__file__))})
    images = dict(evaluation['images']); images[path.relative_to(ROOT).as_posix()] = sha(path)
    write(REPORT/'exports.json', {'images': images,
        'original_GT_receipt_sha256': sha(SOURCE/'dataset.json'),
        'original_GT_receipt_matches_preexisting_D5_freeze': True,
        'all_original_dataset_hashes_verified': True, 'all192_independent_aggregate_replay_passed': True,
        'source_sha256': sha(Path(__file__)), 'no_Gaussian_native86_188_PLY_or_video_claim': True})
    print('D7_INDEPENDENT_AGGREGATES_AND_ORIGINAL_GT_PASSED', flush=True)
    for case, values in diagnostics.items():
        print(case, values, flush=True)


if __name__ == '__main__':
    main()
