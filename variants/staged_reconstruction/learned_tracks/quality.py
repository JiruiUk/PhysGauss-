"""Visible image evidence only; never infer material identity from confidence."""
import numpy as np
from ..surface_trajectory.observations import sample, patch_ncc


def qualify(forward, reverse, visible, reverse_visible, masks, grays, config):
    """Reverse time is already flipped back; compare non-query predictions.

    Forward[0] and reverse[-1] are forced query values in the official
    predictor. Neither is counted as a successful prediction. Closure uses
    reverse[0], which is a prediction, against the original forward query.
    """
    forward, reverse = np.asarray(forward), np.asarray(reverse)
    assert forward.shape == reverse.shape and forward.ndim == 3
    length, count, _ = forward.shape
    assert length >= config['min_observations']
    assert visible.shape == reverse_visible.shape == (length, count)
    finite = np.isfinite(forward).all((0, 2)) & np.isfinite(reverse).all((0, 2))
    closure = np.linalg.norm(reverse[0]-forward[0], axis=-1)
    agreement = np.linalg.norm(reverse-forward, axis=-1)
    h, w = masks[0].shape
    inside = ((forward[..., 0] >= 3) & (forward[..., 0] <= w-4)
              & (forward[..., 1] >= 3) & (forward[..., 1] <= h-4)).all(0)
    foreground = np.stack([sample(mask, points) >= .99 for mask, points in zip(masks, forward)]).all(0)
    pair_scores, pair_texture = [], []
    for t in range(1, length):
        ncc, a, b = patch_ncc(grays[t-1], grays[t], forward[t-1], forward[t], config['patch_radius'])
        pair_scores.append(ncc)
        pair_texture.append(np.minimum(a, b))
    ncc = np.min(pair_scores, axis=0)
    texture = np.min(pair_texture, axis=0)
    # Exclude both overwritten query times from direction agreement.
    interior_agreement = agreement[1:-1].max(0)
    checks = {
        'finite': finite, 'inside': inside, 'foreground': foreground,
        'visibility': visible[1:].all(0) & reverse_visible[:-1].all(0),
        'closure': closure <= config['closure_max_px'],
        'direction_agreement': interior_agreement <= config['direction_max_px'],
        'appearance': ncc >= config['ncc_min'], 'texture': texture >= config['patch_std_min'],
    }
    accepted = np.logical_and.reduce(list(checks.values()))
    return accepted, {'closure_px': closure.tolist(), 'interior_agreement_max_px': interior_agreement.tolist(),
                      'minimum_pair_ncc': ncc.tolist(), 'minimum_pair_std': texture.tolist(),
                      'checks': {key: value.tolist() for key, value in checks.items()}}
