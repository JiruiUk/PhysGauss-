"""Known material regions with shared parameters; no learned segmentation."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


class MaterialBank(nn.Module):
    def __init__(self,regions,learn=True):
        super().__init__()
        if not regions or [r['id'] for r in regions]!=list(range(len(regions))):
            raise ValueError('Material IDs must be unique and consecutive from zero')
        if any(r.get('model','neo_hookean')!='neo_hookean' for r in regions):
            raise ValueError('Only Neo-Hookean material regions are supported')
        def raw(key,default):
            values=torch.tensor([r.get(key,default) for r in regions],dtype=torch.float64)-1e-6
            if (values<=0).any():raise ValueError('Material parameters must be positive')
            return values+torch.log(-torch.expm1(-values))
        self.raw_mu=nn.Parameter(raw('mu_init',1.),requires_grad=learn)
        self.raw_kappa=nn.Parameter(raw('kappa_init',10.),requires_grad=learn)

    def values(self):
        return {'mu':functional.softplus(self.raw_mu)+1e-6,
                'kappa':functional.softplus(self.raw_kappa)+1e-6}

    def forward(self,material_ids):
        if material_ids.dtype!=torch.long:
            raise ValueError('Material IDs must be integer tensors')
        if material_ids.numel() and (material_ids.min()<0 or material_ids.max()>=len(self.raw_mu)):
            raise ValueError('Material ID outside the shared bank')
        ids=material_ids.to(self.raw_mu.device)
        return {key:value[ids] for key,value in self.values().items()}


def load_material_ids(path,num_materials):
    labels=np.load(Path(path),allow_pickle=False)
    if labels.ndim!=1 or not len(labels) or not np.issubdtype(labels.dtype,np.integer):
        raise ValueError('Known material labels must be a nonempty 1D integer array')
    if labels.min()<0 or labels.max()>=num_materials:
        raise ValueError('Material annotation ID outside the configured bank')
    return torch.from_numpy(labels.astype(np.int64))
