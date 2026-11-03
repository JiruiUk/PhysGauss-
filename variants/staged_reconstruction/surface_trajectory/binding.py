"""Bootstrap-only fixed supports. No future reprojection-error admission."""
import numpy as np
from scipy.optimize import minimize


def fit_weights(homogeneous, targets, initial=None, width=400, height=225):
    """Project the weighted 3D center, not a weighted average of 2D centers."""
    h = np.asarray(homogeneous, np.float64)
    target = np.asarray(targets, np.float64)
    count = h.shape[1]
    initial = np.ones(count)/count if initial is None else np.asarray(initial, np.float64)
    initial = initial/initial.sum()

    def residual(weights):
        predicted = np.einsum('tij,i->tj', h, weights)
        if np.any(predicted[:, 3] <= 0):
            raise ValueError('Bootstrap support behind camera')
        uv = ((predicted[:, :2]/predicted[:, 3:]+1)*[width, height]-1)/2
        return uv-target

    if count == 1:
        weights = np.ones(1)
    else:
        result = minimize(lambda w: np.square(residual(w)).mean(), initial,
            method='SLSQP', bounds=[(0., 1.)]*count,
            constraints={'type': 'eq', 'fun': lambda w: w.sum()-1},
            options={'maxiter': 150, 'ftol': 1e-10})
        if not result.success:
            raise ValueError('Bootstrap weight optimization did not converge: '+result.message)
        weights = np.maximum(result.x, 0)
        weights /= weights.sum()
    return weights, np.linalg.norm(residual(weights), axis=1)


def depth_clusters(depths, tolerance):
    """Complete range check in every bootstrap view; no component/object ID."""
    depths = np.asarray(depths)
    clusters = []
    for row in np.argsort(depths[:, 0], kind='stable'):
        for group in clusters:
            proposed = depths[group+[int(row)]]
            if np.all(np.ptp(proposed, axis=0) <= tolerance):
                group.append(int(row))
                break
        else:
            clusters.append([int(row)])
    return clusters


def candidate_qualified(record, config):
    return (record['valid'] and record['distance'] <= config['candidate_radius_px']
            and record['total_mass'] >= config['minimum_total_mass']
            and record['foreground_mass'] >= config['foreground_purity']*record['total_mass']
            and record['patch_mass'] >= config['minimum_patch_mass'])


def choose_support(observations, targets, config):
    """Only three bootstrap observations are accepted as arguments."""
    if len(observations) != 3 or len(targets) != 3:
        raise ValueError('Exactly three bootstrap observations required')
    rows = sorted(set.intersection(*[set(o) for o in observations]))
    rows = [r for r in rows if all(candidate_qualified(o[r], config) for o in observations)]
    if not rows:
        return None, {'reason': 'No row has qualified visible local support in all three bootstrap views'}
    depths = np.array([[o[r]['depth'] for o in observations] for r in rows])
    clusters = depth_clusters(depths, config['depth_tolerance'])
    hypotheses = []
    for cluster in clusters:
        selected = [rows[i] for i in cluster]
        selected.sort(key=lambda r: (-min(o[r]['patch_mass'] for o in observations), r))
        selected = selected[:config['maximum_support_rows']]
        h = np.array([[o[r]['homogeneous'] for r in selected] for o in observations])
        initial = np.array([min(o[r]['patch_mass'] for o in observations) for r in selected])
        try:
            weights, errors = fit_weights(h, targets, initial)
        except ValueError:
            continue
        hypotheses.append({'rows': selected, 'weights': weights.tolist(),
            'bootstrap_errors_px': errors.tolist(), 'mean_error_px': float(errors.mean()),
            'maximum_error_px': float(errors.max()),
            'depth_range_by_bootstrap': np.ptp([[o[r]['depth'] for r in selected]
                                               for o in observations], axis=1).tolist()})
    hypotheses.sort(key=lambda h: h['mean_error_px'])
    if not hypotheses or hypotheses[0]['maximum_error_px'] > config['bootstrap_max_error_px']:
        return None, {'reason': 'No local depth hypothesis fits bootstrap within the frozen pixel tolerance',
                       'hypotheses': hypotheses}
    gap = None if len(hypotheses) < 2 else hypotheses[1]['mean_error_px']-hypotheses[0]['mean_error_px']
    if (len(hypotheses) > 1 and hypotheses[1]['maximum_error_px'] <= config['bootstrap_max_error_px']
            and gap < config['ambiguity_gap_px']):
        return None, {'reason': 'Multiple depth hypotheses are comparably compatible', 'hypotheses': hypotheses}
    best = hypotheses[0]
    best.update(candidate_rows=len(rows), depth_hypotheses=len(hypotheses), runner_up_gap_px=gap,
                binding_is_conditional=True, physical_material_identity_verified=False)
    return best, None
