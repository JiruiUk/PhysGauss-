"""Shared fixed-k normalized spring strain in world coordinates."""
import math
import torch


def median_rest(lengths):
    """Six fit references -> one fixed median per edge, never a GT rest shape."""
    values = torch.as_tensor(lengths, dtype=torch.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError('Need nonempty fit-frame x edge lengths')
    if not bool(torch.isfinite(values).all() and (values > 1e-8).all()):
        raise ValueError('Degenerate/nonfinite reference remains unknown')
    return torch.quantile(values, .5, dim=0).tolist()


def spring_terms(centers, edges, rest, stiffness=1.):
    if not isinstance(stiffness, (int, float)) or not math.isfinite(stiffness) or stiffness <= 0:
        raise ValueError('Stiffness must be one fixed positive shared scalar')
    zero = centers.sum()*0
    if not edges:
        return dict(energy=zero, strain=centers.new_empty(0), valid=False)
    reference = centers.new_tensor(rest)
    if len(reference) != len(edges) or not bool(torch.isfinite(reference).all() and (reference > 1e-8).all()):
        raise ValueError('Bad rest references')
    ids = torch.as_tensor(edges, dtype=torch.long, device=centers.device)
    lengths = torch.linalg.vector_norm(centers[ids[:, 1]]-centers[ids[:, 0]], dim=-1)
    if not bool(torch.isfinite(lengths).all()):
        raise ValueError('Nonfinite current geometry')
    strain = (lengths-reference)/reference
    energy = stiffness*(torch.sqrt(1+strain.square())-1).mean()
    return dict(energy=energy, strain=strain, valid=True)


def combine(base, terms=None, confidence=1., weight=.01, enabled=False):
    if not enabled:
        return base, base*0
    confidence = torch.as_tensor(confidence, device=base.device, dtype=base.dtype).detach()
    auxiliary = weight*confidence*terms['energy']
    return base+auxiliary, auxiliary
