"""Shared deformation features -> stress, with trajectory-derived velocity."""
import json
import os
from pathlib import Path
import torch
from torch import nn
from variants.flow_comparison.model import configure_from_env as parent_configure


def stress_head(width):
    # Do not consume any global training RNG while extending the architecture.
    saved = torch.random.get_rng_state()
    try:
        head = nn.Linear(width, 6)
    finally:
        torch.random.set_rng_state(saved)
    generator = torch.Generator().manual_seed(20261102)
    with torch.no_grad():
        head.weight.copy_(torch.randn(6, width, generator=generator)*.001)
        head.bias.zero_()
    return head


def configure_from_env(net):
    parent_configure(net)
    path = os.environ.get('PIDG_METRIC_CONFIG')
    config = json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {}
    if config.get('enabled', False) and config.get('stress_source') == 'shared':
        net.requires_grad_(False)
    return net


def configure_deform(net):
    if net.shared_stress is None:
        return
    path = os.environ.get('PIDG_METRIC_CONFIG')
    config = json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {}
    net.shared_stress.requires_grad_(config.get('enabled', False) and config.get('stress_source') == 'shared')


def latent(net, spatial, temporal):
    # Exactly the existing forward trunk. Tests compare with the real warp
    # head's input; this is recomputation with shared parameters, no second grid.
    attention = torch.sigmoid(net.spatial_mlp(spatial))
    if net.directional:
        attention = attention*2-1
    h = net.grid_mlp(net.temporal_mlp(temporal)*attention)
    return net.metric_residual(h) if net.metric_residual is not None else h


def predict_stress(deform, x, t):
    if deform.deform.shared_stress is None:
        raise ValueError('Shared stress head is disabled')
    x = x*deform.scale_xyz[None]
    time = x.new_full((len(x), 1), float(t))
    time = (time*2*deform.grid4d.bound-deform.grid4d.bound)*.9
    spatial, temporal = deform.grid4d(torch.cat([x, time], -1))
    return deform.deform.shared_stress(latent(deform.deform, spatial, temporal))


def optimizer_groups(deform, opt):
    groups = [
        {'params': [p for n, p in deform.deform.named_parameters() if not n.startswith('shared_stress.')],
         'lr': opt.position_lr_init*deform.network_lr_scale, 'name': 'deform'},
        {'params': list(deform.grid4d.parameters()), 'lr': opt.position_lr_init*deform.grid_lr_scale, 'name': 'grid'}]
    if deform.deform.shared_stress is not None:
        groups.append({'params': list(deform.deform.shared_stress.parameters()), 'lr': opt.velocity_net_lr, 'name': 'shared_stress'})
    return groups
