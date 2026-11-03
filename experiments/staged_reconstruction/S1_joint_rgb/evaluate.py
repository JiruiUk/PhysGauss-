"""Independent GT-only evaluator, never imported by the frozen RGB trainer."""
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from variants.staged_reconstruction.prepare_data import ROOT, read, sha
from variants.staged_reconstruction.joint_rgb.io import BASE, REPORT, RUNS, OUT, atomic_json, verify, stopped, lock
from variants.staged_reconstruction.joint_rgb.core import assert_equivalent
from variants.staged_reconstruction.train_static import setup, render_static, PIPE
from variants.staged_reconstruction.nuisance_static.runtime import calibrated_camera
from variants.staged_reconstruction.nuisance_static.evaluate import image_metrics, panel
from variants.staged_reconstruction.adaptive_static.prepare import world_points
from variants.staged_reconstruction.synthetic_audit.evaluate import score
from variants.staged_reconstruction.synthetic_audit.input import load


def cpu_rgb(package):
    return package['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def ray_counts(package, gt):
    alpha = package['alpha'].squeeze().cpu().numpy()
    depth = package['depth'].squeeze().cpu().numpy()/np.maximum(alpha, 1e-8)
    surface = gt['static_depth'] > 0; thin = gt['thin_static_visible'] & surface
    valid = (alpha > .5) & np.isfinite(depth)
    good = valid & (np.abs(depth-gt['static_depth']) <= .05)
    return {'static_GT_pixels': int(surface.sum()), 'thin_GT_pixels': int(thin.sum()),
        'static_valid_pixels': int((valid & surface).sum()), 'thin_valid_pixels': int((valid & thin).sum()),
        'accurate_static_ray_pixels': int((good & surface).sum()),
        'accurate_thin_ray_pixels': int((good & thin).sum()),
        'static_missing_pixels': int((surface & ~valid).sum()),
        'thin_missing_pixels': int((thin & ~valid).sum())}


def audit_checkpoint(case, arm, protocol):
    directory = RUNS/'formal'/case/arm
    assert sha(directory/'checkpoint.pt') == read(directory/'checkpoint.json')['sha256']
    final = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
    initial = torch.load(directory/'initial.pt', map_location='cpu', weights_only=False)
    parent = torch.load(ROOT/protocol['parents'][case]['checkpoint'], map_location='cpu', weights_only=False)
    assert final['iteration'] == protocol['updates'] and len(final['sample_trace']) == protocol['updates']
    assert final['source_sha256'] == sha(REPORT/'source_freeze.json')
    assert final['protocol_sha256'] == sha(BASE/'protocol.json')
    assert final['parent_sha256'] == protocol['parents'][case]['sha256']
    rng = random.Random(); rng.setstate(parent['rng_python']); stack = list(parent['camera_stack']); expected = []
    for _ in range(protocol['updates']):
        if not stack:
            stack = rng.sample(range(96), 96)
        expected.append(stack.pop())
    assert final['sample_trace'] == expected and final['camera_stack'] == stack
    assert final['rng_python'] == rng.getstate()
    assert final['visits'] == np.bincount(expected, minlength=96).tolist()
    assert final['case'] == case and final['arm'] == arm
    geometry = ['_xyz', '_scaling', '_rotation', 'point_ids', 'born_at', 'initial_xyz']
    names = list(parent['model']) if arm == 'frozen_static' else geometry
    for name in names:
        assert torch.equal(final['model']['static.'+name], parent['model'][name]), name
    for i, group in enumerate(parent['optimizer']['param_groups']):
        assert group == final['optimizers'][0]['param_groups'][i]
        if arm == 'frozen_static' or group['name'] in geometry:
            parameter_id = group['params'][0]
            assert_equivalent(parent['optimizer']['state'][parameter_id], final['optimizers'][0]['state'][parameter_id], 0, 0)
    for name, value in final['model'].items():
        assert torch.isfinite(value).all(), name
    for name in ['static.point_ids', 'static.born_at', 'dynamic.point_ids', 'dynamic.born_at']:
        assert torch.equal(final['model'][name], initial['model'][name]), name
    result = {'passed': True, 'case': case, 'arm': arm, 'iteration': final['iteration'],
        'static_and_dynamic_rows': [4000, 4000], 'full_6000_sample_trace_and_parent_RNG_replay': True,
        'static_geometry_and_frozen_moments_exact': True, 'source_protocol_parent_verified': True,
        'initial_sha256': sha(directory/'initial.pt'), 'checkpoint_sha256': sha(directory/'checkpoint.pt')}
    atomic_json(REPORT/f'formal_{case}_{arm}_audit.json', result)
    return final, result


def aggregate(rows, arm):
    result = {}
    for key in rows[0]['models'][arm]:
        if key == 'ray_counts':
            totals = {k: sum(row['models'][arm][key][k] for row in rows) for k in rows[0]['models'][arm][key]}
            totals['accurate_static_ray_coverage'] = totals['accurate_static_ray_pixels']/max(totals['static_GT_pixels'], 1)
            totals['thin_alpha_recall_full_denominator'] = totals['thin_valid_pixels']/max(totals['thin_GT_pixels'], 1)
            result[key] = totals
        else:
            values = [row['models'][arm][key] for row in rows if row['models'][arm][key] is not None]
            result[key] = float(np.mean(values)) if values else None
    return result


@torch.no_grad()
def main():
    assert not stopped() and not (REPORT/'synthetic_evaluation.json').exists()
    verify(); assert read(REPORT/'recovery.json')['passed']
    torch.set_num_threads(4); cv2.setNumThreads(2); renderer = setup()
    from variants.staged_reconstruction.joint_rgb.runtime import JointScene
    protocol = read(BASE/'protocol.json'); OUT.mkdir(parents=True, exist_ok=True)
    receipt = read(ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2/dataset.json')
    for name, expected in receipt['files'].items():
        assert sha(ROOT/name) == expected, name
    results, gates, bindings, images = {}, {}, {}, {}
    with lock('evaluation.lock'):
        for case in protocol['cases']:
            models, states = {}, {}
            for arm in protocol['arms']:
                states[arm], bindings[case+'/'+arm] = audit_checkpoint(case, arm, protocol)
                scene = JointScene(protocol, case, arm == 'revisable_static')
                scene.load_state_dict(states[arm]['model']); models[arm] = scene
            assert states['frozen_static']['sample_trace'] == states['revisable_static']['sample_trace']
            private = ROOT/protocol['synthetic_data']/case/'evaluation_only'
            meta = read(private/'cameras.json'); k = np.asarray(meta['K'])
            static_rows, composite_rows, references, thin_references, sheets = [], [], [], [], []
            for row in meta['frames']:
                if stopped():
                    raise RuntimeError('User stop: preserve partial evaluation')
                camera = calibrated_camera(np.asarray(row['world_to_camera']), k, meta['width'], meta['height'])
                with np.load(private/'gt'/f"c{row['camera']:02d}_t{row['time_index']}.npz") as source:
                    gt = {name: source[name].copy() for name in source.files}
                composite = {'camera': row['camera'], 'time_index': row['time_index'],
                    'novel_camera': not row['seen_camera'], 'models': {}}
                # First4 are preregistered primary times. The dataset also has
                # t4; retain it separately rather than silently dropping it.
                for arm, model in models.items():
                    package = model.render(renderer, camera, row['time'])
                    mse = float(np.mean((cpu_rgb(package)-gt['observed_rgb'])**2))
                    composite['models'][arm] = {'composite_observed_PSNR': float(-10*np.log10(max(mse, 1e-12)))}
                composite_rows.append(composite)
                if row['time_index'] != 0:
                    continue
                static_row = {'camera': row['camera'], 'novel_camera': not row['seen_camera'], 'models': {}}
                yy, xx = np.nonzero(gt['static_depth'] > 0)
                xyz = world_points(np.c_[xx, yy], gt['static_depth'][yy, xx], np.asarray(row['world_to_camera']), k)
                references.append(xyz); thin_references.append(xyz[gt['thin_static_visible'][yy, xx]])
                values = [('Evaluator-only static GT', gt['static_rgb'])]
                for arm, model in models.items():
                    package = render_static(renderer, camera, model.static)
                    metrics, rgb = image_metrics(package, gt, np.asarray(row['world_to_camera']), k)
                    metrics['ray_counts'] = ray_counts(package, gt); static_row['models'][arm] = metrics
                    values.append((arm+' static', rgb))
                values.append(('Observed RGB includes motion', gt['observed_rgb']))
                if row['camera'] in [0, 8, 24, 31]:
                    sheets.append(panel(values, f"{case} c{row['camera']}"))
                static_rows.append(static_row)
            def unique(x):
                _, indices = np.unique(np.floor(x/.015).astype(int), axis=0, return_index=True)
                return x[indices]
            reference = unique(np.concatenate(references)); thin = unique(np.concatenate(thin_references))
            aggregates = {}
            for arm, scene in models.items():
                active = (scene.static.get_opacity[:, 0] >= .1).cpu().numpy()
                geom = score(scene.static.get_xyz.cpu().numpy()[active], reference, thin, .05)
                subset = [r for r in static_rows if r['novel_camera']]
                assert len(static_rows) == 32 and len(subset) == 8
                primary = [r['models'][arm]['composite_observed_PSNR'] for r in composite_rows
                    if r['novel_camera'] and r['time_index'] in protocol['fixed_time_indices']]
                all_times = [r['models'][arm]['composite_observed_PSNR'] for r in composite_rows if r['novel_camera']]
                assert len(primary) == 32 and len(all_times) == 40
                aggregates[arm] = {'all_static_cameras': aggregate(static_rows, arm), 'novel_static_cameras': aggregate(subset, arm),
                    'geometry_active_opacity_ge_01': geom, 'active_static_rows': int(active.sum()),
                    'low_opacity_static_rows_retained': int((~active).sum()),
                    'novel_composite_PSNR_first4_primary_times': float(np.mean(primary)),
                    'novel_composite_PSNR_all5_times_supplementary': float(np.mean(all_times))}
            control, candidate = [aggregates[arm] for arm in protocol['arms']]
            c, n = control['novel_static_cameras'], candidate['novel_static_cameras']
            cg, ng = control['geometry_active_opacity_ge_01'], candidate['geometry_active_opacity_ge_01']
            limits, extra = protocol['synthetic_promotion_gate'], protocol['additional_gates']
            delta = {'static_clean_GT_PSNR_gain_dB': n['clean_GT_PSNR']-c['clean_GT_PSNR'],
                'static_RGB_spill_reduction_fraction': 1-n['static_RGB_spill']/max(c['static_RGB_spill'], 1e-12),
                'visible_surface_coverage_loss_percentage_points': 100*(cg['surface_coverage']-ng['surface_coverage']),
                'thin_surface_coverage_loss_percentage_points': 100*(cg['thin_surface_coverage']-ng['thin_surface_coverage']),
                'novel_composite_PSNR_degradation_dB': control['novel_composite_PSNR_first4_primary_times']-candidate['novel_composite_PSNR_first4_primary_times'],
                'accurate_static_ray_coverage_loss_percentage_points': 100*(c['ray_counts']['accurate_static_ray_coverage']-n['ray_counts']['accurate_static_ray_coverage']),
                'thin_alpha_recall_loss_percentage_points': 100*(c['ray_counts']['thin_alpha_recall_full_denominator']-n['ray_counts']['thin_alpha_recall_full_denominator'])}
            checks = {'static_clean_GT_PSNR': delta['static_clean_GT_PSNR_gain_dB'] >= limits['static_clean_GT_PSNR_gain_min_dB'],
                'spill': delta['static_RGB_spill_reduction_fraction'] >= limits['static_RGB_spill_reduction_min_fraction'],
                'surface': delta['visible_surface_coverage_loss_percentage_points'] <= limits['visible_surface_coverage_loss_max_percentage_points'],
                'thin_surface': delta['thin_surface_coverage_loss_percentage_points'] <= limits['thin_surface_coverage_loss_max_percentage_points'],
                'dynamic_generalization': delta['novel_composite_PSNR_degradation_dB'] <= extra['novel_composite_PSNR_degradation_max_dB'],
                'accurate_rays': delta['accurate_static_ray_coverage_loss_percentage_points'] <= extra['accurate_static_ray_coverage_loss_max_percentage_points'],
                'thin_alpha': delta['thin_alpha_recall_loss_percentage_points'] <= extra['thin_alpha_recall_loss_max_percentage_points'],
                'nonempty_support': ng['points'] > 0 and n['ray_counts']['static_valid_pixels'] > 0 and n['ray_counts']['thin_valid_pixels'] > 0}
            gates[case] = {'numerical_passed': bool(all(checks.values())), 'checks': {key: bool(value) for key, value in checks.items()}, 'deltas': delta}
            image_path = OUT/f'{case}_static_comparison.png'; assert not image_path.exists()
            Image.fromarray(np.concatenate(sheets)).save(image_path); images[image_path.relative_to(ROOT).as_posix()] = sha(image_path)
            training = load(ROOT/protocol['synthetic_data']/case/'train'); decompositions = []
            for index in protocol['fixed_train_indices']:
                camera = calibrated_camera(training['views'][index], training['K'], meta['width'], meta['height'])
                for arm, model in models.items():
                    static = render_static(renderer, camera, model.static)
                    dynamic = renderer(camera, model.at(training['times'][index]), PIPE, model.center.new_zeros(3), 0., 0., 0.)
                    combined = model.render(renderer, camera, training['times'][index])
                    decompositions.append(panel([('Observed training RGB', training['rgb'][index]), ('static', cpu_rgb(static)),
                        ('shared dynamic pool', cpu_rgb(dynamic)), ('combined', cpu_rgb(combined))], f'{case} {arm} train{index}'))
            image_path = OUT/f'{case}_train_decomposition.png'; assert not image_path.exists()
            Image.fromarray(np.concatenate(decompositions)).save(image_path); images[image_path.relative_to(ROOT).as_posix()] = sha(image_path)
            results[case] = {'aggregate': aggregates, 'static_frames': static_rows, 'composite_frames': composite_rows,
                'geometry_contract': 'opacity>=0.1 centers,0.05 to visible GT; not thin thickness or ellipsoid shape',
                'depth_contract': 'nativeD/alpha; missing and accurate rays use full independent static/thin GT denominator',
                'time_contract': 'first4 fixed primary times; dataset fifth time retained separately; unchanged preregistered gate'}
            print('J1_EVALUATED', case, gates[case], flush=True)
        verify()
        atomic_json(REPORT/'synthetic_evaluation.json', {'results': results, 'checkpoint_audits': bindings,
            'images': images, 'GT_used_by_trainer': False, 'dataset_receipt_sha256': sha(ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2/dataset.json'),
            'all_original_data_hashes_verified': True, 'evaluation_source_sha256': sha(Path(__file__))})
        atomic_json(REPORT/'synthetic_gate.json', {'numerical_passed': all(g['numerical_passed'] for g in gates.values()),
            'cases': gates, 'manual_review_required': True, 'passed': False, 'real_training_authorized': False})


if __name__ == '__main__':
    main()
