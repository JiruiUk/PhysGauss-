"""Camera-invariant local support geometry, not physical thickness ground truth."""
import torch
import torch.nn.functional as F


def covariance(scales, quaternions):
    q = F.normalize(quaternions, dim=-1, eps=1e-12)
    w, x, y, z = q.unbind(-1)
    rotation = torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                            2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                            2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)
    return (rotation * scales.square().unsqueeze(-2)) @ rotation.transpose(-1, -2)


def support_spectrum(centers, covariances):
    delta = centers-centers.mean(0)
    moment = (covariances+delta[..., None]*delta[:, None, :]).mean(0)
    return torch.linalg.eigvalsh(moment).clamp_min(1e-12).sqrt()


def interval(value, reference, lower, upper):
    ratio = value/reference.clamp_min(1e-12)
    return F.relu(lower-ratio).square()+F.relu(ratio-upper).square()


def geometry_terms(centers, covariances, edges, reference, config):
    """All gradients use current geometry. Parent references are disclosed priors."""
    zero = centers.sum()*0+covariances.sum()*0
    spectrum = support_spectrum(centers, covariances)
    ref_spectrum = centers.new_tensor(reference['spectrum'])
    low, high = config['lower_ratio'], config['upper_ratio']
    thickness = interval(spectrum[0], ref_spectrum[0], low, high)
    area = interval(spectrum[1]*spectrum[2], ref_spectrum[1]*ref_spectrum[2], low, high)
    current_max = torch.linalg.eigvalsh(covariances).clamp_min(1e-12)[..., -1].sqrt()
    cap = F.relu(current_max/centers.new_tensor(reference['max_std']).clamp_min(1e-12)-high).square().mean()
    connection = zero
    for (i, j), old_length in zip(edges, reference['edge_lengths']):
        delta = centers[j]-centers[i]
        length = torch.linalg.vector_norm(delta)
        direction = delta/length.clamp_min(1e-8)
        radii = config['sigma_support'] * sum((direction @ covariances[k] @ direction).clamp_min(1e-12).sqrt() for k in [i, j])
        gap = F.relu((length-radii)/max(old_length, 1e-8)).square()
        collapse = F.relu(config['edge_lower_ratio']-length/max(old_length, 1e-8)).square()
        connection = connection+gap+collapse
    return dict(connection=connection/max(len(edges), 1), thickness=thickness,
                area_scale=area+cap, spectrum=spectrum)


def footprint(field, uv, radius=2):
    """Bilinear native contribution samples at the actual observed RGB track."""
    h, w = field.shape
    offsets = field.new_tensor([(i, j) for j in range(-radius, radius+1) for i in range(-radius, radius+1)])
    points = offsets+field.new_tensor(uv)
    if not bool(torch.isfinite(points).all() and (points[:, 0]>=0).all() and
                (points[:, 0]<=w-1).all() and (points[:, 1]>=0).all() and (points[:, 1]<=h-1).all()):
        return None
    grid = points/field.new_tensor([w-1, h-1])*2-1
    return F.grid_sample(field[None, None], grid[None, None], mode='bilinear',
                         padding_mode='zeros', align_corners=True).mean()


def coverage_term(contribution, observed_uv, parent_mean, lower=.8):
    value = footprint(contribution, observed_uv)
    if value is None or parent_mean is None or parent_mean<=1e-8:
        return contribution.sum()*0, False
    return F.relu(lower-value/parent_mean).square(), True


def combine(base, terms, confidence=1., weight=.01, enabled=False):
    if not enabled:
        return base, base*0
    auxiliary = weight*confidence*sum(terms.values())/4
    return base+auxiliary, auxiliary
