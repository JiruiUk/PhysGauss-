"""Independent static/visible-thin GT audit; no changes to inference or gates."""
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ..prepare_data import ROOT, read, write, sha
from ..pretrained_depth.evaluate import describe, heat
from .prepare import BASE, REPORT, RUNS, SOURCE, verify, stopped


def main():
    assert not stopped(); verify(); assert not (REPORT/'evaluation.json').exists()
    receipt = read(REPORT/'inference.json'); protocol = read(BASE/'protocol.json')
    assert read(REPORT/'input_audit.json')['passed']
    for path, h in receipt['outputs'].items(): assert sha(ROOT/path) == h, path
    for path, h in read(SOURCE/'dataset.json')['files'].items(): assert sha(ROOT/path) == h, path
    output = ROOT/'output/staged_reconstruction/S1_rgb_plane_warp/v1'; output.mkdir(parents=True, exist_ok=True)
    results, gates, figures = {}, {}, {}
    for case in protocol['synthetic_cases']:
        meta = read(ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train/cameras.json')
        buckets = {arm: {'counts': Counter(), 'static_error': [], 'visible_error': [], 'thin_error': []} for arm in ['D1', 'D6']}
        changes = Counter(); changed_errors = {'old': [], 'new': []}; rows, panels = [], []
        for i, row in enumerate(meta['frames']):
            assert not stopped()
            with np.load(RUNS/case/f'view_{i:03d}.npz') as d:
                uv, old, new, accepted = d['uv'], d['original'], d['updated'], d['accepted']
                eligible = d['eligible']; slope_admitted = d['slope_admitted']
                assert np.array_equal(new[~accepted], old[~accepted]) and np.array_equal(new != old, accepted)
            yy, xx = uv[:, 1], uv[:, 0]
            with np.load(ROOT/f'data/derived/clean_static_synthetic/v2/{case}/evaluation_only/gt'/(Path(row['image']).stem+'.npz')) as gt:
                target = gt['static_depth'][yy, xx]; surface = target > 0
                full_visible = gt['observed_static_visible'] & (gt['static_depth'] > 0)
                full_thin = gt['thin_static_visible'] & full_visible
                visible, thin = full_visible[yy, xx], full_thin[yy, xx]
                rgb = (gt['observed_rgb']*255).round().astype('uint8')
            for arm, values in [('D1', old), ('D6', new)]:
                error = abs(values-target); bucket = buckets[arm]
                c = {'candidates': len(values), 'on_static': int(surface.sum()), 'gross_or_no_static': int((~surface | (error > .05)).sum()),
                    'visible_total': int(full_visible.sum()), 'thin_total': int(full_thin.sum()),
                    'accurate_visible_005': int((visible&(error <= .05)).sum()), 'accurate_thin_002': int((thin&(error <= .02)).sum())}
                bucket['counts'].update(c); bucket['static_error'].append(error[surface])
                bucket['visible_error'].append(error[visible]); bucket['thin_error'].append(error[thin])
            old_error, new_error = abs(old-target), abs(new-target)
            on = accepted & surface
            change = {'accepted': int(accepted.sum()), 'eligible': int(eligible.sum()), 'slope_admitted': int(slope_admitted.sum()),
                'accepted_on_static': int(on.sum()), 'accepted_no_static': int((accepted&~surface).sum()),
                'static_error_improved': int((on&(new_error < old_error)).sum()),
                'static_error_worsened': int((on&(new_error > old_error)).sum()),
                'gross_corrected': int((on&(old_error > .05)&(new_error <= .05)).sum()),
                'gross_introduced': int((on&(old_error <= .05)&(new_error > .05)).sum())}
            changes.update(change); changed_errors['old'].append(old_error[on]); changed_errors['new'].append(new_error[on])
            rows.append({'image': row['image'], **change})
            if i in [0, 24, 72]:
                height, width = meta['height'], meta['width']
                def ray_map(values, mask):
                    z = np.zeros((height, width), np.float32); v = np.zeros((height, width), bool)
                    z[yy, xx] = values; v[yy, xx] = mask
                    return heat(z, v, .1)
                change_map = np.zeros_like(rgb); change_map[yy[eligible], xx[eligible]] = [80, 80, 80]
                better = on & (new_error < old_error); worse = on & (new_error > old_error)
                change_map[yy[better], xx[better]] = [30, 220, 70]; change_map[yy[worse], xx[worse]] = [240, 50, 60]
                tiles = [('Input RGB', rgb), ('D1 static error 0..0.10', ray_map(old_error, surface)),
                    ('D6 SAME-pixel static error', ray_map(new_error, surface)), ('Eligible gray; changed GT +/-', change_map)]
                canvas = Image.new('RGB', (width*4, height+36), '#171923'); draw = ImageDraw.Draw(canvas)
                for j, (label, image) in enumerate(tiles):
                    canvas.paste(Image.fromarray(image), (width*j, 36)); draw.text((width*j+2, 2), case+' '+Path(row['image']).stem, fill='white')
                    draw.text((width*j+2, 18), label, fill='white')
                panels.append(np.asarray(canvas))
        for arm, bucket in buckets.items():
            c = bucket['counts']; ratio = lambda a, b: c[a]/max(c[b], 1)
            bucket['counts'] = dict(c)
            for key in ['static_error', 'visible_error', 'thin_error']: bucket[key] = describe(np.concatenate(bucket[key]))
            bucket['metrics'] = {'supported_static_depth_MAE': bucket['static_error']['mean'],
                'gross_error_gt_005_or_no_static_fraction': ratio('gross_or_no_static', 'candidates'),
                'accurate_visible_static_ray_005_coverage': ratio('accurate_visible_005', 'visible_total'),
                'accurate_visible_thin_ray_002_coverage': ratio('accurate_thin_002', 'thin_total')}
        old, new = buckets['D1'], buckets['D6']; limit = protocol['numeric_improvement_gate_both_cases']
        failed = []
        if new['static_error']['mean'] > old['static_error']['mean'] * limit['all_supported_static_depth_MAE_relative_max']: failed.append('static_MAE_5percent_improvement')
        if new['metrics']['gross_error_gt_005_or_no_static_fraction'] > old['metrics']['gross_error_gt_005_or_no_static_fraction'] * limit['gross_error_or_no_static_fraction_relative_max']: failed.append('gross_error_5percent_improvement')
        for key in ['mean', 'p90']:
            if new['thin_error'][key] is None or new['thin_error'][key] > old['thin_error'][key]: failed.append('thin_'+key+'_no_increase')
        for key in ['accurate_visible_static_ray_005_coverage', 'accurate_visible_thin_ray_002_coverage']:
            if new['metrics'][key] < old['metrics'][key]: failed.append(key+'_no_decrease')
        strict = protocol['follow_on_Gaussian_protocol_evidence_gate_both_cases']; strict_failed = []
        for name, bound in strict.items():
            if name.endswith('_max') and new['metrics'][name[:-4]] > bound: strict_failed.append(name[:-4])
            if name.endswith('_min') and new['metrics'][name[:-4]] < bound: strict_failed.append(name[:-4])
        gates[case] = {'numeric_improvement_passed': not failed, 'failed_numeric_requirements': failed,
            'Gaussian_evidence_passed': not strict_failed, 'failed_Gaussian_requirements': strict_failed}
        results[case] = {'arms': buckets, 'changes': dict(changes), 'accepted_depth_error': {k: describe(np.concatenate(v)) for k, v in changed_errors.items()},
            'membership_and_unaccepted_depths_exact': True, 'frames': rows}
        path = output/f'{case}_RGB_warp.png'; assert not path.exists()
        Image.fromarray(np.concatenate(panels)).save(path); figures[path.relative_to(ROOT).as_posix()] = sha(path)
        print('RGB_PLANE_WARP_GT_AUDIT', case, {arm: b['metrics'] for arm, b in buckets.items()}, dict(changes), gates[case], flush=True)
    verify()
    for path, h in receipt['outputs'].items(): assert sha(ROOT/path) == h, path
    write(REPORT/'evaluation.json', {'results': results, 'images': figures, 'independent_GT_evaluation': True,
        'GT_not_used_for_proposals_acceptance_or_selection': True, 'Gaussian_training': False, 'inference_sha256': sha(REPORT/'inference.json')})
    write(REPORT/'gate.json', {'both_cases_numeric_improvement': all(g['numeric_improvement_passed'] for g in gates.values()),
        'both_cases_Gaussian_evidence': all(g['Gaussian_evidence_passed'] for g in gates.values()), 'cases': gates,
        'Gaussian_training_started': False, 'real_training_authorized': False,
        'separate_paired_Gaussian_protocol_and_full_native_recovery_still_required': True})


if __name__ == '__main__': main()
