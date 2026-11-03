"""Compare frozen wide-pair predictions with static and known-camera scales."""
import argparse
from pathlib import Path

import numpy as np

from ..prepare_data import ROOT, DATA, read, write, sha
from .prepare import EXP
from .infer import verify
from .core import calibrate_scale, projection_errors


def quantiles(values):
    if not len(values):
        return None
    # Behind-camera points remain infinite failures rather than being dropped.
    values = np.sort(np.asarray(values, dtype=np.float64))
    result = []
    for q in [.5, .9]:
        index = q * (len(values)-1)
        lo, hi = int(np.floor(index)), int(np.ceil(index))
        value = values[lo] if lo == hi else (1-index+lo)*values[lo] + (index-lo)*values[hi]
        result.append(float(value) if np.isfinite(value) else None)
    return result


def below(value, index, maximum):
    return value is not None and value[index] is not None and value[index] <= maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-calibrated-pointmap-pilot', action='store_true')
    args = parser.parse_args()
    if not args.enable_calibrated_pointmap_pilot:
        raise SystemExit('Calibrated pointmap qualification is opt-in')
    assert not (EXP / 'qualification.json').exists()
    import torch
    from scipy.ndimage import binary_erosion
    from ..motion_integrity.prepare import load_parent
    from ..stage2.train import image
    from ..train_static import camera, PIPE
    from ..union.core import dynamic_mask
    from output.short_clip_diagnostic.camera_math import intrinsic, w2c

    torch.set_num_threads(4)
    plan = verify(); gates = plan['old_pilot_gates_unchanged']
    inference = read(EXP / 'inference.json')
    assert inference['protocol_sha256'] == sha(EXP / 'protocol.json')
    assert inference['source_freeze_sha256'] == sha(EXP / 'source_freeze.json')
    meta = read(DATA / 'transforms_train.json')
    frames = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    renderer, scene, _, _, _ = load_parent()
    k = intrinsic(meta['camera_angle_x'], 'metadata', 512, 288)
    yy, xx = np.indices((288, 512)); pixels = np.stack([xx, yy], -1)
    fit_lattice = (yy + 2*xx) % 3 == 0
    cache = {}; context = {}; rows = []

    def reference_for(n):
        if n not in cache:
            rgba = image(frames[n], 512)
            r, g, b = rgba[..., :3].transpose(2, 0, 1)
            static = binary_erosion((rgba[..., 3] > .95) & (b > r*1.3+.02) & (b > g*1.05), iterations=3)
            with torch.no_grad():
                package = renderer(camera(frames[n], meta, 512), scene.static, PIPE,
                                   scene.centers.new_zeros(3), 0., 0., 0.)
            alpha = package['alpha'].squeeze().cpu().numpy()
            reference = package['depth'].squeeze().cpu().numpy() / np.maximum(alpha, 1e-6)
            static &= (alpha > .99) & (reference > 1) & (reference < 15)
            cache[n] = {'rgba': rgba, 'static': static,
                        'dynamic': binary_erosion(dynamic_mask(rgba), iterations=2), 'depth': reference}
        return cache[n]

    for item in inference['pairs']:
        anchor, partner = item['anchor'], item['partner']
        path = ROOT / item['path']; assert sha(path) == item['sha256']
        for n in [anchor, partner]:
            assert sha(DATA / (frames[n]['file_path'] + '.png')) == item['source_RGB_sha256'][str(n)]
        ref = reference_for(anchor)
        second_rgba = image(frames[partner], 512)
        with np.load(path) as data:
            first = data['first_points'].astype(np.float64)
            second = data['second_points_in_first'].astype(np.float64)
            confidence, second_confidence = data['first_confidence'], data['second_confidence']
        assert first.shape == second.shape == (288, 512, 3)
        assert confidence.shape == second_confidence.shape == (288, 512)
        assert all(np.isfinite(a).all() for a in [first, second, confidence, second_confidence])
        raw = first[..., 2]
        support = ref['static'] & (raw > 1e-5) & (confidence >= gates['dynamic_confidence_min'])
        ys, xs = np.nonzero(support)
        fit = np.arange(len(xs)) % 3 == 0; held = ~fit
        dm = ref['dynamic'] & (raw > 1e-5) & (confidence >= gates['dynamic_confidence_min'])
        fraction = float(dm.sum() / max(1, ref['dynamic'].sum()))
        raw_projection = quantiles(projection_errors(first[dm], pixels[dm], k, np.eye(3), np.zeros(3), 1.))
        if support.any() and dm.any():
            lo, hi = np.quantile(raw[support], [.01, .99])
            extrap = float(((raw[dm] < lo) | (raw[dm] > hi)).mean())
        else:
            extrap = None
        control_scale = float(np.median(ref['depth'][ys[fit], xs[fit]] / raw[ys[fit], xs[fit]])) if len(xs) > 100 else None
        second_support = binary_erosion(second_rgba[..., 3] > .95, iterations=2)
        second_support &= second_confidence >= gates['dynamic_confidence_min']
        camera_fit, camera_held = second_support & fit_lattice, second_support & ~fit_lattice
        relative = w2c(frames[partner]['transform_matrix'], 'metadata_cv') @ np.linalg.inv(w2c(frames[anchor]['transform_matrix'], 'metadata_cv'))
        calibration = calibrate_scale(second[camera_fit], pixels[camera_fit], k, relative[:3, :3], relative[:3, 3],
            minimum_support=plan['scale_fit_min'], cauchy_relative_baseline=plan['robust_cauchy_relative_baseline'], iterations=plan['IRLS_iterations'])
        camera_scale = calibration['scale']
        camera_error = quantiles(projection_errors(second[camera_held], pixels[camera_held], k,
            relative[:3, :3], relative[:3, 3], camera_scale)) if camera_scale is not None else None
        camera_checks = {
            'camera_scalar_observable': bool(calibration['scale_observable']),
            'camera_fit_support': int(camera_fit.sum()) >= plan['scale_fit_min'],
            'camera_held_support': int(camera_held.sum()) >= plan['scale_held_min'],
            'camera_predicted_parallax': calibration.get('median_predicted_ray_parallax_degrees', 0.) >= plan['median_predicted_ray_parallax_min_degrees'],
            'camera_held_projection_median': below(camera_error, 0, plan['held_second_projection_median_max_pixels']),
            'camera_held_projection_p90': below(camera_error, 1, plan['held_second_projection_p90_max_pixels'])}

        for arm, scale in zip(plan['arms'], [control_scale, camera_scale]):
            depth = raw * scale if scale is not None else None
            static_error = quantiles(np.abs(depth[ys[held], xs[held]]-ref['depth'][ys[held], xs[held]])) if depth is not None else None
            checks = {
                'positive_scalar': scale is not None and np.isfinite(scale) and scale > 0,
                'static_support': len(xs) >= gates['static_support_min'],
                'held_static_median': below(static_error, 0, gates['held_static_median_max']),
                'held_static_p90': below(static_error, 1, gates['held_static_p90_max']),
                'dynamic_confidence': fraction >= gates['dynamic_confident_fraction_min'],
                'dynamic_extrapolation': extrap is not None and extrap <= gates['dynamic_extrapolation_fraction_max'],
                'raw_first_projection_median': below(raw_projection, 0, gates['dynamic_self_projection_median_max_pixels']),
                'raw_first_projection_p90': below(raw_projection, 1, gates['dynamic_self_projection_p90_max_pixels'])}
            if arm == plan['arms'][1]:
                checks.update(camera_checks)
            checks = {name: bool(value) for name, value in checks.items()}
            result = {'arm': arm, 'anchor': anchor, 'partner': partner, 'actual_frame_gap': abs(partner-anchor),
                'scale': scale, 'static_support': len(xs), 'static_fit_support': int(fit.sum()), 'static_held_support': int(held.sum()),
                'static_depth_error_median_p90': static_error, 'static_reference_is_model_conditioned_not_GT': True,
                'dynamic_confident_fraction': fraction, 'dynamic_extrapolation_fraction': extrap,
                'raw_first_projection_error_median_p90_pixels': raw_projection,
                'checks': checks, 'passed': all(checks.values()), 'failed_checks': [n for n, v in checks.items() if not v]}
            if arm == plan['arms'][1]:
                result.update(camera_calibration=calibration, second_held_support=int(camera_held.sum()),
                              second_held_projection_error_median_p90_pixels=camera_error)
            rows.append(result)
            context.setdefault((arm, anchor), []).append((depth, dm))
            print('CALIBRATED_POINTMAP_QUALIFIED', arm, anchor, partner, result, flush=True)

    stability = []
    for arm in plan['arms']:
        for anchor in plan['anchors']:
            values = context[(arm, anchor)]
            assert len(values) == 2
            (a, ma), (b, mb) = values
            common = ma & mb
            coverage = float(common.sum() / max(1, cache[anchor]['dynamic'].sum()))
            error = quantiles(np.abs(a[common]-b[common])) if a is not None and b is not None else None
            checks = {'both_scales_available': a is not None and b is not None,
                'common_dynamic_coverage': coverage >= gates['dynamic_confident_fraction_min'],
                'partner_depth_median': below(error, 0, gates['dynamic_partner_depth_median_max']),
                'partner_depth_p90': below(error, 1, gates['dynamic_partner_depth_p90_max'])}
            stability.append({'arm': arm, 'anchor': anchor, 'common_dynamic_fraction': coverage,
                'partner_depth_difference_median_p90': error, 'checks': checks, 'passed': bool(all(checks.values()))})
    verify()
    write(EXP / 'qualification.json', {'pair_rows': rows, 'partner_stability': stability,
        'arm_passed': {arm: all(r['passed'] for r in rows+stability if r['arm'] == arm) for arm in plan['arms']},
        'protocol_sha256': sha(EXP / 'protocol.json'), 'inference_sha256': sha(EXP / 'inference.json'),
        'source_freeze_sha256': sha(EXP / 'source_freeze.json'), 'static_reference_is_model_conditioned_not_GT': True,
        'dynamic_geometry_truth_validated': False, 'training_admitted': False,
        'PIDG_optimizer_updates': 0, 'test_or_val_RGB_used': False,
        'null_metric_means_unavailable_or_nonfinite_not_a_pass': True})


if __name__ == '__main__':
    main()
