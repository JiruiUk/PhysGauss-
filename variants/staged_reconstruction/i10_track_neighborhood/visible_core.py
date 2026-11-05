"""Select persistent rows using split-local observed compositing support."""
import numpy as np


def footprint_ids(ids, uv, static_count, population, radius=2):
    ids = np.asarray(ids); x, y = np.floor(uv).astype(int)
    h, w = ids.shape[1:]
    if x-radius < 0 or y-radius < 0 or x+radius+1 >= w or y+radius+1 >= h:
        return []
    values = np.unique(ids[:, y-radius:y+radius+2, x-radius:x+radius+2])
    return (values[(values >= static_count) & (values < static_count+population)]-static_count).tolist()


def choose(uv, xyz, projected, depth, candidates, observations, fit, held, config):
    errors = np.linalg.norm(projected-np.asarray(uv)[:, None], axis=-1)
    scores = np.sqrt(np.mean(errors[fit]**2, axis=0))
    eligible = []
    for row in candidates:
        good = sum(observations[int(row)][i]['passed'] for i in fit)
        if (good >= 4 and np.isfinite(xyz[:, row]).all() and np.isfinite(errors[:, row]).all()
                and (depth[:, row] > .01).all() and scores[row] <= 2.):
            eligible.append(int(row))
    eligible.sort(key=lambda row: (scores[row], row))
    if not eligible:
        return dict(accepted=False, reason='no_split_visible_geometric_candidate', eligible_rows=[])
    row = eligible[0]
    rivals = [other for other in eligible[1:] if np.linalg.norm(xyz[len(xyz)//2, row]-xyz[len(xyz)//2, other]) >= .05]
    alternative = min((float(scores[r]) for r in rivals), default=None)
    margin = alternative-float(scores[row]) if alternative is not None else None
    held_rms = float(np.sqrt(np.mean(errors[held, row]**2)))
    checks = dict(fit=bool(scores[row] <= 2.), held=held_rms <= 2., maximum=bool(errors[:, row].max() <= 4.),
                  alternative=bool(margin is None or margin >= .25))
    return dict(accepted=all(checks.values()), row=row, fit_rms_px=float(scores[row]), held_rms_px=held_rms,
                max_px=float(errors[:, row].max()), alternative_fit_rms_px=alternative, margin_px=margin,
                checks=checks, eligible_rows=eligible, unique_only_within_bounded_visible_set=True,
                world_depth_is_independent_GT=False)


def reciprocal_result(a, b, xyz, projected, config):
    if 'row' not in a or 'row' not in b:
        return dict(passed=False, reason='missing_split_candidate')
    i, j = a['row'], b['row']
    distance = float(np.linalg.norm(xyz[:, i]-xyz[:, j], axis=-1).max())
    pixels = float(np.linalg.norm(projected[:, i]-projected[:, j], axis=-1).max())
    return dict(passed=bool(a['accepted'] and b['accepted'] and distance <= config['equivalent_max_world_distance']
                           and pixels <= config['equivalent_max_projection_distance_px']),
                same_persistent_ID=i == j, max_world_separation=distance, max_projected_separation_px=pixels,
                surface_equivalence_is_model_conditioned=True)
