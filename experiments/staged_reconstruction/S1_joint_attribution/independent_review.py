"""Independent all-camera replay without importing the measured accounting/mixer."""
from pathlib import Path
import numpy as np
import torch
from variants.staged_reconstruction.prepare_data import ROOT, read, sha
from variants.staged_reconstruction.joint_rgb.io import atomic_json
from variants.staged_reconstruction.train_static import setup, render_static
from variants.staged_reconstruction.adaptive_static.model import AdaptiveGaussian
from variants.staged_reconstruction.nuisance_static.runtime import calibrated_camera


@torch.no_grad()
def main():
    base = Path(__file__).resolve().parent; report = base/'v1'
    assert not (report/'independent_audit.json').exists()
    manifest = read(report/'source_freeze.json')
    for name, expected in manifest.items(): assert sha(ROOT/name) == expected, name
    measured = read(report/'measurement.json'); protocol = read(base/'protocol.json')
    old_base = ROOT/protocol['accepted_base']; old_protocol = read(old_base/'protocol.json')
    runs = ROOT/protocol['accepted_runs']; renderer = setup(); torch.set_num_threads(4)
    assert not any(p.exists() for p in [ROOT/'stop.request', runs/'stop.request', runs.parent/'stop.request',
        ROOT/'runs/pidg_staged_reconstruction/stop.request',
        ROOT/'runs/pidg_staged_reconstruction/S1_joint_attribution/stop.request',
        ROOT/'runs/pidg_staged_reconstruction/S1_joint_attribution/v1/stop.request'])
    cases = {}
    for case in protocol['cases']:
        states = [torch.load(runs/'formal'/case/arm/'checkpoint.pt', map_location='cpu', weights_only=False)['model']
            for arm in ['frozen_static', 'revisable_static']]
        private = ROOT/protocol['data']/case/'evaluation_only'; meta = read(private/'cameras.json')
        tally = {subset: {g: np.zeros((3, 3), dtype=np.int64) for g in ['surface', 'thin']} for subset in ['all32', 'novel8']}
        replay_rows = 0
        for row in meta['frames']:
            if row['time_index'] != 0: continue
            gt = np.load(private/'gt'/f"c{row['camera']:02d}_t0.npz")
            ground_depth, ground_rgb = gt['static_depth'], gt['static_rgb']
            masks = {'surface': ground_depth > 0, 'thin': gt['thin_static_visible'] & (ground_depth > 0)}
            camera = calibrated_camera(np.asarray(row['world_to_camera']), np.asarray(meta['K']), meta['width'], meta['height'])
            ref = next(r for r in measured['results'][case]['frames'] if r['camera'] == row['camera'])
            rendered, classes = {}, {}
            for name in ['pp', 'cp', 'pc', 'cc']:
                payload = {}
                for key, value in states[0].items():
                    if not key.startswith('static.'): continue
                    field = key[7:]
                    choice = name[0] if field == '_opacity' else name[1] if field in ['_features_dc', '_features_rest'] else 'p'
                    payload[field] = states[int(choice == 'c')][key]
                model = AdaptiveGaussian(ROOT/old_protocol['parents'][case]['initialization'])
                model.load_payload(payload); model.requires_grad_(False)
                package = render_static(renderer, camera, model)
                rgb = package['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                alpha = package['alpha'].squeeze().cpu().numpy(); d = package['depth'].squeeze().cpu().numpy()
                rendered[name] = {'rgb': rgb, 'alpha': alpha, 'D': d}
                depth = np.divide(d, np.maximum(alpha, 1e-8))
                valid = np.logical_and(alpha > .5, np.isfinite(depth))
                good = np.logical_and(valid, np.abs(depth-ground_depth) <= .05)
                classes[name] = np.where(good, 0, np.where(valid, 1, 2))
                mse = float(np.mean(np.square(rgb-ground_rgb)))
                spill = float(np.mean(np.abs(rgb[~masks['surface']])))
                assert mse == ref['RGB'][name]['MSE'] and spill == ref['RGB'][name]['static_RGB_spill']
            for a, b in [('pp', 'pc'), ('cp', 'cc')]:
                assert all(np.array_equal(rendered[a][k], rendered[b][k]) for k in ['alpha', 'D'])
            if row['camera'] == 24:
                for replicate in ['a', 'b']:
                    with np.load(ROOT/'runs/pidg_staged_reconstruction/S1_joint_attribution/v1'/f'{case}_{replicate}.npz') as cache:
                        assert all(np.array_equal(cache[f'{s}_{k}'], v) for s, items in rendered.items() for k, v in items.items())
            for group, mask in masks.items():
                matrix = np.zeros((3, 3), dtype=np.int64)
                for i in range(3):
                    for j in range(3): matrix[i, j] = np.count_nonzero(mask & (classes['pp'] == i) & (classes['cc'] == j))
                assert matrix.tolist() == ref['transitions'][group]['matrix_old_rows_new_columns']
                tally['all32'][group] += matrix
                if not row['seen_camera']: tally['novel8'][group] += matrix
            gt.close(); replay_rows += 1
        for subset, groups in tally.items():
            for group, matrix in groups.items():
                recorded = measured['results'][case][subset][group]
                assert matrix.tolist() == recorded['matrix_old_rows_new_columns']
                assert int(matrix.sum()) == recorded['denominator']
                assert int(matrix[:, 0].sum()-matrix[0].sum()) == recorded['net_accurate_change']
            for metric, result in measured['results'][case][subset]['RGB_attribution'].items():
                x = measured['results'][case][subset]['RGB_metrics_mean_per_camera']
                opacity = ((x['cp'][metric]-x['pp'][metric])+(x['cc'][metric]-x['pc'][metric]))/2
                sh = ((x['pc'][metric]-x['pp'][metric])+(x['cc'][metric]-x['cp'][metric]))/2
                assert opacity == result['opacity_symmetric_marginal_change']
                assert sh == result['SH_symmetric_marginal_change']
        cases[case] = {'all32_native_four_state_replays': replay_rows, 'full_denominators_and_transition_counts_exact': True,
            'all_per_camera_RGB_MSE_spill_exact': True, 'c24_matches_both_independent_processes': True,
            'SH_only_depth_alpha_exact': True, 'symmetric_marginal_equations_match': True}
        print('J2_INDEPENDENT_REVIEW', case, replay_rows, flush=True)
    for name, expected in manifest.items(): assert sha(ROOT/name) == expected, name
    atomic_json(report/'independent_audit.json', {'passed': True, 'cases': cases,
        'files_verified_before_and_after': len(manifest), 'measurement_sha256': sha(report/'measurement.json'),
        'review_source_sha256': sha(Path(__file__)), 'training_updates': 0})


if __name__ == '__main__': main()
