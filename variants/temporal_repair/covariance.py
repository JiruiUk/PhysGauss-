"""Opt-in unit rotations for newly trained covariance models; default is exact identity."""
import torch
from variants.color_repair.color import policy


def unit_quaternion(raw,enabled=False):
    if not enabled:return raw
    norm=raw.norm(dim=-1,keepdim=True)
    identity=torch.zeros_like(raw);identity[...,0]=1.
    return torch.where(norm>1e-8,raw/norm.clamp_min(1e-8),identity)


def rotation_for_render(raw):
    config=policy()
    return unit_quaternion(raw,config.get('enabled',False) and config.get('integrity_unit_rotation',False))
