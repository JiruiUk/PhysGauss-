"""Explicitly opt-in clock control for diagnostic copies only."""
import torch
from torch import nn


class RawTimeFeatures(nn.Module):
    def __init__(self,frequencies):
        super().__init__()
        # Retain checkpoint key/shape; these values do not drive the fixed clock.
        self.register_buffer('frequencies',frequencies.detach().clone())

    def forward(self,t):
        t=t.reshape(-1,1)
        return torch.cat([t,t.new_zeros((len(t),15))],-1)


def apply_clock(net,mode='legacy'):
    if mode=='legacy':return net
    if mode!='fixed_monotonic':raise ValueError('Unknown diagnostic clock mode')
    net.time_embed=RawTimeFeatures(net.time_embed.frequencies)
    with torch.no_grad():
        net.time_proj.weight.zero_();net.time_proj.weight[0,0]=2.
        net.time_proj.bias.fill_(-1.)
    for p in net.time_proj.parameters():p.requires_grad_(False)
    return net
