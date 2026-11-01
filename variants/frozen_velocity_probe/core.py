"""Frozen-target regression and explicit alignment metrics."""
import hashlib
import torch


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def velocity_parameters(module):
    # This is a diagnostic copy. Shared features may change; the resulting stress
    # output is NOT claimed to preserve the original stress and is never deployed.
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(not name.startswith('stress_head.'))
    return [p for p in module.parameters() if p.requires_grad]


def normalized_mse(prediction, target, scale):
    if target.requires_grad:
        raise ValueError('Targets must be detached from the reconstruction')
    return ((prediction-target)/scale).square().mean()


@torch.no_grad()
def alignment_metrics(prediction, target, moving_threshold):
    if prediction.shape != target.shape or target.shape[-1] != 3:
        raise ValueError('Expected equally shaped 3D vectors')
    p, d = prediction.reshape(-1, 3).double(), target.reshape(-1, 3).double()
    error = (p-d).square().mean().sqrt()
    rms = d.square().mean().sqrt()
    pn, dn = p.norm(dim=-1), d.norm(dim=-1)
    moving = dn > moving_threshold
    directional = moving & (pn > max(1e-10, moving_threshold*1e-4))
    angles = torch.rad2deg(torch.acos(((p[directional]*d[directional]).sum(-1)/
                                     (pn[directional]*dn[directional])).clamp(-1, 1)))
    ratios = pn[moving]/dn[moving]
    result = dict(count=len(p), rmse=error.item(), target_rms=rms.item(),
                  predicted_rms=p.square().mean().sqrt().item(),
                  relative_rmse=(error/rms.clamp_min(1e-12)).item(),
                  moving_count=moving.sum().item(), direction_count=directional.sum().item(),
                  angle_median_deg=angles.median().item() if len(angles) else None,
                  angle_p90_deg=torch.quantile(angles, .9).item() if len(angles) else None,
                  speed_ratio_median=ratios.median().item() if len(ratios) else None,
                  moving_relative_rmse=(((p[moving]-d[moving]).square().mean().sqrt()/
                                        d[moving].square().mean().sqrt()).item()) if moving.any() else None)
    return result


def disabled(config):
    # The disabled command returns before importing model code, loading a source
    # checkpoint, drawing randomness, creating an optimizer or touching CUDA.
    return not config.get('enabled', False)


def partition_candidates(candidates, particle_ids, ntrain, nhold, seed):
    """Partition cloning families first so held-out embeddings cannot be updated."""
    generator=torch.Generator(device='cpu').manual_seed(seed)
    unique=particle_ids[candidates].unique()
    ordered=unique[torch.randperm(len(unique),generator=generator).to(unique.device)]
    cut=max(1,min(len(ordered)-1,int(.8*len(ordered))))
    known=candidates[torch.isin(particle_ids[candidates],ordered[:cut])]
    held=candidates[torch.isin(particle_ids[candidates],ordered[cut:])]
    if len(known)<ntrain or len(held)<nhold:
        raise ValueError('Insufficient candidate rows after family-disjoint split')
    a=known[torch.randperm(len(known),generator=generator)[:ntrain].to(known.device)]
    b=held[torch.randperm(len(held),generator=generator)[:nhold].to(held.device)]
    assert not torch.isin(particle_ids[a],particle_ids[b]).any()
    return torch.cat([a,b])
