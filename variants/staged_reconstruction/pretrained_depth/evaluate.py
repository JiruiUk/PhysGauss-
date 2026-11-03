"""Independent GT audit of fixed observed-depth prior; never writes training data."""
from pathlib import Path
from itertools import combinations
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
import cv2
from ..prepare_data import ROOT, read, write, sha
from ..synthetic_audit.generate import trace
from .prepare import BASE, REPORT, RUNS, SOURCE, verify, stopped


def describe(values):
    v = np.asarray(values)
    return {'n': int(v.size), 'mean': float(v.mean()) if v.size else None,
            'median': float(np.median(v)) if v.size else None,
            'p90': float(np.quantile(v, .9)) if v.size else None}


def heat(values, valid, maximum):
    finite = np.nan_to_num(values, nan=0., posinf=maximum, neginf=0.)
    rgb = cv2.cvtColor(cv2.applyColorMap((np.clip(finite / maximum, 0, 1)*255).astype('uint8'),
                                       cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb


def main():
    assert not stopped()
    verify(); assert not (REPORT / 'evaluation.json').exists(), 'Preserve original evaluation'
    receipt = read(REPORT / 'inference.json'); p = read(BASE / 'protocol.json')
    assert read(REPORT / 'replay.json')['passed']
    for name, h in receipt['outputs'].items(): assert sha(ROOT / name) == h, name
    for name, h in read(SOURCE / 'dataset.json')['files'].items(): assert sha(ROOT / name) == h, name
    results, gates, pictures = {}, {}, {}
    output = ROOT / 'output/staged_reconstruction/S1_pretrained_geometry/v1'
    output.mkdir(parents=True, exist_ok=True)
    for case in p['synthetic_cases']:
        meta = read(ROOT / f'data/derived/clean_static_synthetic/v2/{case}/train/cameras.json')
        k = np.asarray(meta['K']); height, width = meta['height'], meta['width']
        counts = Counter(); errors = {key: [] for key in ['visible', 'thin', 'observed', 'same_D1_visible', 'D1_visible', 'temporal']}
        rows, panels, views, relative_rank = [], [], [], []
        reason = Counter()
        for i, row in enumerate(meta['frames']):
            assert not stopped()
            info = read(RUNS / case / f'view_{i:03d}.json'); reason[info['reason']] += 1
            with np.load(RUNS / case / f'view_{i:03d}.npz') as data:
                raw, depth = data['raw_relative'], data['aligned_depth']
            gt_path = ROOT / f'data/derived/clean_static_synthetic/v2/{case}/evaluation_only/gt' / (Path(row['image']).stem + '.npz')
            with np.load(gt_path) as gt:
                target = gt['static_depth'].copy(); surface = target > 0
                visible = gt['observed_static_visible'] & surface
                thin = gt['thin_static_visible'] & visible
                observed = trace(np.asarray(row['world_to_camera']), k, width, height, row['time'], case == 'glossy')
                assert np.array_equal(observed['rgb'].astype(np.float32), gt['observed_rgb'])
                assert np.array_equal(observed['static_visible'], gt['observed_static_visible'])
                rgb = (gt['observed_rgb']*255).round().astype('uint8')
            valid = depth > 0; empty = observed['depth'] == 0; hidden = surface & ~visible
            error = abs(depth-target)
            c = {'visible_total': int(visible.sum()), 'visible_valid': int((valid&visible).sum()),
                 'visible_good_005': int((valid&visible&(error <= .05)).sum()),
                 'visible_bad_005': int((valid&visible&(error > .05)).sum()),
                 'thin_total': int(thin.sum()), 'thin_valid': int((thin&valid).sum()),
                 'thin_good_002': int((valid&thin&(error <= .02)).sum()),
                 'empty_total': int(empty.sum()), 'empty_valid': int((empty&valid).sum()),
                 'hidden_total': int(hidden.sum()), 'hidden_valid': int((hidden&valid).sum()),
                 'all_valid': int(valid.sum()), 'all_pixels': valid.size,
                 'observed_total': int((observed['depth'] > 0).sum())}
            counts.update(c)
            errors['visible'].append(error[valid&visible]); errors['thin'].append(error[valid&thin])
            errors['observed'].append(abs(depth-observed['depth'])[valid&(observed['depth'] > 0)])
            with np.load(ROOT / f'runs/pidg_staged_reconstruction/S1_depth_refinement/v1/{case}/view_{i:03d}.npz') as d1:
                same = (d1['refined'] > 0) & visible & valid
                errors['same_D1_visible'].append(error[same]); errors['D1_visible'].append(abs(d1['refined']-target)[same])
            rank = float(spearmanr(raw[visible], 1./target[visible]).statistic) if visible.sum() > 1 else None
            relative_rank.append(rank)
            rows.append({'image': row['image'], 'calibrated': info['calibrated'], 'reason': info['reason'],
                         'held_MAE': info.get('held_MAE'), 'raw_static_inverse_depth_spearman': rank, **c})
            views.append({'depth': depth, 'visible': visible, 'pose': np.asarray(row['world_to_camera'])})
            if i in [0, 24, 72]:
                raw_vis = heat(raw - raw.min(), np.ones(raw.shape, bool), max(float(np.ptp(raw)), 1e-8))
                classes = np.zeros_like(rgb); classes[visible] = [60, 60, 60]
                classes[valid&visible&(error <= .05)] = [40, 220, 80]
                classes[valid&visible&(error > .05)] = [250, 50, 60]
                classes[valid&empty] = [110, 65, 220]; classes[valid&hidden] = [240, 170, 30]
                canvas = Image.new('RGB', (width*5, height+42), '#171923'); draw = ImageDraw.Draw(canvas)
                tiles = [('Input RGB', rgb), ('Raw relative, per-view display', raw_vis),
                         ('Train-anchor metric depth 0..7', heat(depth, valid, 7)),
                         ('Static abs error 0..0.10', heat(error, valid&visible, .1)),
                         ('GT diag: green/red; BG purple', classes)]
                for j, (label, tile) in enumerate(tiles):
                    canvas.paste(Image.fromarray(tile), (j*width, 42))
                    draw.text((j*width+2, 2), f'{case} {Path(row["image"]).stem}', fill='white')
                    draw.text((j*width+2, 15), label, fill='white')
                    draw.text((j*width+2, 28), 'calibrated' if info['calibrated'] else info['reason'], fill='white')
                panels.append(np.asarray(canvas))
        for a, b in combinations(range(len(views)), 2):
            va, vb = views[a], views[b]
            if not np.array_equal(va['pose'], vb['pose']): continue
            visible = va['visible'] & vb['visible']
            both = visible & (va['depth'] > 0) & (vb['depth'] > 0)
            counts['temporal_eligible'] += int(visible.sum()); counts['temporal_valid'] += int(both.sum())
            errors['temporal'].append(abs(va['depth']-vb['depth'])[both])
        cat = lambda key: np.concatenate(errors[key]) if errors[key] else np.array([])
        ratio = lambda a, b: counts[a] / max(counts[b], 1)
        metrics = {'calibrated_view_fraction': receipt['results'][case]['calibrated_views']/96,
                   'calibrated_views': receipt['results'][case]['calibrated_views'],
                   'accurate_visible_static_ray_005_coverage': ratio('visible_good_005', 'visible_total'),
                   'accurate_thin_static_ray_002_coverage': ratio('thin_good_002', 'thin_total'),
                   'visible_static_depth_MAE': describe(cat('visible'))['mean'],
                   'visible_static_error_gt_005_fraction': ratio('visible_bad_005', 'visible_valid'),
                   'empty_background_predicted_valid_fraction': ratio('empty_valid', 'empty_total'),
                   'hidden_static_behind_dynamic_predicted_valid_fraction': ratio('hidden_valid', 'hidden_total'),
                   'thin_depth_P90': describe(cat('thin'))['p90']}
        failed = []
        for name, bound in p['independent_evidence_gate_both_cases_required'].items():
            if name == 'nonempty_all_required': continue
            key, direction = name.rsplit('_', 1); value = metrics[key]
            if value is None or (value < bound if direction == 'min' else value > bound): failed.append(key)
        if not all(counts[key] for key in ['visible_valid', 'thin_valid', 'empty_total', 'hidden_total']): failed.append('empty_evaluation_support')
        gates[case] = {'passed': not failed, 'failed_requirements': failed}
        results[case] = {'metrics': metrics, 'counts': dict(counts), 'reasons': dict(reason),
                         'error_diagnostics': {key: describe(cat(key)) for key in errors},
                         'raw_static_inverse_depth_rank': describe(np.asarray([v for v in relative_rank if v is not None and np.isfinite(v)])),
                         'temporal_comparable_fraction': ratio('temporal_valid', 'temporal_eligible'), 'frames': rows}
        image_path = output / f'{case}_prior_audit.png'; assert not image_path.exists()
        Image.fromarray(np.concatenate(panels)).save(image_path); pictures[image_path.relative_to(ROOT).as_posix()] = sha(image_path)
        print('PRETRAINED_DEPTH_GT_AUDIT', case, metrics, gates[case], flush=True)
    verify()
    for name, h in receipt['outputs'].items(): assert sha(ROOT / name) == h, name
    write(REPORT / 'evaluation.json', {'results': results, 'images': pictures, 'independent_GT_evaluation': True,
          'GT_not_used_for_inference_alignment_or_selection': True, 'observed_raytracer_replay_matches_saved_GT': True,
          'Gaussian_training': False, 'inference_sha256': sha(REPORT / 'inference.json')})
    write(REPORT / 'gate.json', {'both_cases_evidence_passed': all(g['passed'] for g in gates.values()), 'cases': gates,
          'Gaussian_training_started': False, 'real_training_authorized': False,
          'gate_tests_direct_static_use_not_general_observed_depth_utility': True,
          'next_requires_separate_paired_Gaussian_protocol_and_native_recovery': True})
    print('PRETRAINED_DEPTH_EVALUATED', gates, flush=True)


if __name__ == '__main__':
    main()
