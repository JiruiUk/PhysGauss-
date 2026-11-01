"""Keep every original parameter and optimizer group for exact full-state resume."""
import json
import os
from pathlib import Path
import torch


def configure_clock(net,mode='legacy'):
    if mode=='legacy':return net
    if mode!='fixed_monotonic':raise ValueError('Unsupported velocity_clock_mode')
    net.velocity_clock_mode=mode
    # Keep Parameter objects, state_dict keys and optimizer groups intact. Only
    # their unused legacy clock gradients are disabled; the fixed clock uses t.
    for module in (net.time_embed,net.time_proj):
        for p in module.parameters():p.requires_grad_(False)
    return net


def configure_from_env(net):
    path=os.environ.get('PIDG_METRIC_CONFIG')
    if not path:return net
    config=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not config.get('enabled',False):return net
    return configure_clock(net,config.get('velocity_clock_mode','legacy'))


def time_coordinate(net,t):
    if getattr(net,'velocity_clock_mode','legacy')=='fixed_monotonic':
        return torch.tanh(2*t-1).squeeze(-1)
    # Exactly the original operations and graph in the default mode.
    return torch.tanh(net.time_proj(net.time_embed(t))).squeeze(-1)
