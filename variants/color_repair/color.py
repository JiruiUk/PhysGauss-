"""Opt-in Python SH colors at the exact deformed CUDA viewing directions."""
from functools import lru_cache
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from variants.color_repair.model import positive_color


@lru_cache(maxsize=8)
def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def policy():
    path=os.environ.get('PIDG_METRIC_CONFIG')
    return _read(path) if path else {}


def enabled():
    c=policy()
    return c.get('enabled',False) and c.get('positive_color',False)


def raw_sh_color(pc,means,camera):
    from utils.sh_utils import eval_sh
    directions=F.normalize(means-camera.camera_center.reshape(1,3),dim=-1,eps=1e-12)
    coefficients=pc.get_features.transpose(1,2)
    return eval_sh(pc.active_sh_degree,coefficients,directions)+.5


def colors(pc,means,camera):
    return positive_color(raw_sh_color(pc,means,camera),float(policy().get('color_beta',5.)))
