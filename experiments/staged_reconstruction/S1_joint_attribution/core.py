"""Independent frozen-output accounting; no trainer, renderer or GT asset exporter."""
import numpy as np

STATES = ('pp', 'cp', 'pc', 'cc')
APPEARANCE = {'_opacity', '_features_dc', '_features_rest'}


def mixed_payload(parent, candidate, name):
    assert name in STATES and parent.keys() == candidate.keys()
    for key in parent.keys() - APPEARANCE:
        assert np.array_equal(parent[key].cpu().numpy(), candidate[key].cpu().numpy()), key
    result = {k: v.clone() for k, v in parent.items()}
    for key in APPEARANCE:
        source = candidate if name[0 if key == '_opacity' else 1] == 'c' else parent
        result[key] = source[key].clone()
    return result


def labels(alpha, accumulated_depth, gt_depth):
    depth = accumulated_depth / np.maximum(alpha, 1e-8)
    valid = (alpha > .5) & np.isfinite(depth)
    result = np.full(alpha.shape, 2, dtype=np.uint8)
    result[valid] = 1
    result[valid & (np.abs(depth-gt_depth) <= .05)] = 0
    return result, depth


def transitions(old, new, surface):
    a, b = old[surface], new[surface]
    matrix = np.bincount(3*a.astype(np.int64)+b, minlength=9).reshape(3, 3)
    return summarize(matrix)


def summarize(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    old, new = matrix.sum(1), matrix.sum(0)
    lost, gained = int(matrix[0, 1:].sum()), int(matrix[1:, 0].sum())
    assert int(new[0]-old[0]) == gained-lost
    return {'matrix_old_rows_new_columns': matrix.tolist(), 'denominator': int(matrix.sum()),
        'old_counts': old.tolist(), 'new_counts': new.tolist(),
        'lost_accurate': lost, 'gained_accurate': gained, 'net_accurate_change': gained-lost,
        'lost_to_valid_inaccurate': int(matrix[0, 1]), 'lost_to_missing': int(matrix[0, 2]),
        'old_accurate_fraction': float(old[0]/max(matrix.sum(), 1)),
        'new_accurate_fraction': float(new[0]/max(matrix.sum(), 1))}


def shift_sums(old_label, new_label, old_depth, new_depth, gt_depth, surface):
    common = surface & (old_label != 2) & (new_label != 2)
    lost_valid = surface & (old_label == 0) & (new_label == 1)
    result = {}
    for name, mask in [('common_valid', common), ('lost_to_valid_inaccurate', lost_valid)]:
        shift = new_depth[mask].astype(np.float64)-old_depth[mask]
        old_error = np.abs(old_depth[mask].astype(np.float64)-gt_depth[mask])
        new_error = np.abs(new_depth[mask].astype(np.float64)-gt_depth[mask])
        result[name] = {'n': int(mask.sum()), 'signed_shift_sum': float(shift.sum()),
            'absolute_shift_sum': float(np.abs(shift).sum()),
            'old_abs_error_sum': float(old_error.sum()), 'new_abs_error_sum': float(new_error.sum())}
    return result


def attribution(values):
    pp, cp, pc, cc = [values[k] for k in STATES]
    opacity = .5*((cp-pp)+(cc-pc)); sh = .5*((pc-pp)+(cc-cp))
    assert np.isclose(opacity+sh, cc-pp, atol=1e-12)
    return {'total_change': cc-pp, 'opacity_symmetric_marginal_change': opacity,
        'SH_symmetric_marginal_change': sh, 'interaction': cc-cp-pc+pp,
        'fixed_render_intervention_not_training_causality': True}


def aggregate(rows):
    result = {}
    for group in ('surface', 'thin'):
        matrix = sum((np.asarray(r['transitions'][group]['matrix_old_rows_new_columns']) for r in rows),
            np.zeros((3, 3), dtype=np.int64))
        result[group] = summarize(matrix)
        for subset in ('common_valid', 'lost_to_valid_inaccurate'):
            sums = {key: sum(r['depth_shifts'][group][subset][key] for r in rows)
                for key in rows[0]['depth_shifts'][group][subset]}
            means = {key.removesuffix('_sum')+'_mean': value/max(sums['n'], 1)
                for key, value in sums.items() if key != 'n'}
            result[group][subset] = {**sums, **means}
    result['RGB_metrics_mean_per_camera'] = {state: {key: float(np.mean([r['RGB'][state][key] for r in rows]))
        for key in rows[0]['RGB'][state]} for state in STATES}
    result['RGB_attribution'] = {key: attribution({s: result['RGB_metrics_mean_per_camera'][s][key] for s in STATES})
        for key in ('MSE', 'static_RGB_spill')}
    result['cameras'] = len(rows)
    return result
