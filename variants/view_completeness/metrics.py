"""Separate target-grounded coverage from target-free shape proxies.

No metric here establishes 3D geometry, material identity or true visibility.
The optional entry point is inert before inspecting inputs when disabled.
"""
import numpy as np
from scipy import ndimage


def _alpha(value):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 5:
        raise ValueError('Expected a 2D alpha image with both dimensions >=5')
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
        raise ValueError('Alpha must be finite and in [0,1]')
    return array


def shape_proxies(alpha, threshold=.5, thin_radius=2, island_pixels=20, hole_pixels=4):
    """Descriptive proxies at a DECLARED resolution, not a completeness score.

    Opening and filling are used only to measure the original mask. Neither
    operation modifies a model render or supplies a pseudo training target.
    Empty predictions have undefined ratios, never a perfect score.
    """
    a = _alpha(alpha)
    if not 0 < threshold < 1 or thin_radius < 1 or island_pixels < 1 or hole_pixels < 1:
        raise ValueError('Invalid fixed diagnostic thresholds')
    mask = a > threshold
    area = int(mask.sum())
    common = {'foreground_pixels': area, 'foreground_fraction': float(mask.mean()),
              'touches_image_border': bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
              'target_free_proxy_only': True, 'physical_completeness_verified': False}
    if not area:
        return {**common, 'defined': False, 'component_count': 0,
                'small_island_fraction': None, 'thin_support_fraction': None,
                'enclosed_hole_fraction': None, 'boundary_complexity': None}
    structure = np.ones((3, 3), bool)
    labels, count = ndimage.label(mask, structure)
    sizes = np.bincount(labels.ravel())[1:]
    small = int(sizes[sizes < island_pixels].sum())
    # Pad with foreground continuation at image edges: frame clipping alone
    # should not manufacture a thin feature along the border.
    padded = np.pad(mask, thin_radius+1, mode='edge')
    yy, xx = np.mgrid[-thin_radius:thin_radius+1, -thin_radius:thin_radius+1]
    disk = xx*xx+yy*yy <= thin_radius*thin_radius
    opened = ndimage.binary_opening(padded, structure=disk)
    pad = thin_radius+1
    opened = opened[pad:-pad, pad:-pad]
    holes = ndimage.binary_fill_holes(mask) & ~mask
    hole_labels, _ = ndimage.label(holes, structure)
    hole_sizes = np.bincount(hole_labels.ravel())[1:]
    hole_area = int(hole_sizes[hole_sizes >= hole_pixels].sum())
    boundary = mask & ~ndimage.binary_erosion(np.pad(mask, 1, mode='edge'), structure=structure)[1:-1, 1:-1]
    # A geometric descriptor, sensitive to legitimate silhouettes as well as artifacts.
    complexity = float(boundary.sum()**2/(4*np.pi*area))
    return {**common, 'defined': True, 'component_count': int(count),
            'small_island_fraction': small/area,
            'thin_support_fraction': float((mask & ~opened).sum()/area),
            'enclosed_hole_fraction': hole_area/(area+hole_area),
            'boundary_complexity': complexity}


def observed_coverage(alpha, target):
    """Only valid for the SAME time AND camera as an independent alpha target."""
    a, t = _alpha(alpha), _alpha(target)
    if a.shape != t.shape:
        raise ValueError('Alpha and true observation must share pixel coordinates')
    foreground, predicted = t > .5, a > .5
    background = ~foreground
    core = ndimage.minimum_filter(t, size=7, mode='constant', cval=0) > .95
    def rate(event, domain):
        return float((event & domain).sum()/domain.sum()) if domain.any() else None
    union = foreground | predicted
    return {'foreground_missing_fraction': rate(~predicted, foreground),
            'core_alpha_below_09_fraction': rate(a < .9, core),
            'background_spill_fraction': rate(predicted, background),
            'mask_IoU': float((foreground & predicted).sum()/union.sum()) if union.any() else None}


def evaluate(alpha=None, target=None, *, enabled=False, **settings):
    if not enabled:
        return None
    return {'shape_proxies': shape_proxies(alpha, **settings),
            'independent_alpha_target_available': target is not None,
            'observed_coverage': observed_coverage(alpha, target) if target is not None else None,
            'full_object_completeness_score': None}


def summarize(rows):
    """Keep mean, p95 and maximum separate; never average away bad views."""
    result = {'samples': len(rows), 'undefined_shape_samples': sum(not r['shape_proxies']['defined'] for r in rows)}
    for field in ['small_island_fraction', 'thin_support_fraction', 'enclosed_hole_fraction', 'boundary_complexity']:
        pairs = [(r['shape_proxies'][field], r['frame']) for r in rows if r['shape_proxies'][field] is not None]
        if not pairs:
            result[field] = None
            continue
        values = np.asarray([p[0] for p in pairs])
        worst = max(pairs, key=lambda p: p[0])
        result[field] = {'mean': float(values.mean()), 'p95': float(np.quantile(values, .95)),
                         'maximum': float(worst[0]), 'maximum_frame': worst[1]}
    return result
