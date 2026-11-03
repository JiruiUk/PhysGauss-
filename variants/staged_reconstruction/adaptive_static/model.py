"""Independent topology changes preserving existing native Gaussian/Adam rows."""
import numpy as np
import torch
from torch import nn
from ..static_gaussian import StaticGaussian


class AdaptiveGaussian(StaticGaussian):
    def __init__(self, initialization, device='cuda'):
        super().__init__(initialization, device=device)
        with np.load(initialization, allow_pickle=False) as data:
            ids = torch.tensor(data['point_ids'], device=device, dtype=torch.long)
        self.register_buffer('point_ids', ids)
        self.register_buffer('born_at', torch.zeros_like(ids))
        with torch.no_grad(): self._opacity.fill_(float(np.log(.1/.9)))

    def load_payload(self, payload):
        device = self._xyz.device
        for name, _ in list(self.named_parameters()):
            setattr(self, name, nn.Parameter(payload[name].to(device).clone()))
        for name in ['initial_xyz', 'point_ids', 'born_at']:
            setattr(self, name, payload[name].to(device).clone())


@torch.no_grad()
def replace_rows(model, optimizer, keep, extra=None):
    """Preserve old Adam step and selected moments; new rows start with zero moments."""
    device = model._xyz.device
    keep = torch.as_tensor(keep, device=device, dtype=torch.bool)
    assert keep.shape == (len(model._xyz),) and keep.any()
    extra = extra or {}
    for group in optimizer.param_groups:
        name = group['name']; old = group['params'][0]
        assert old is getattr(model, name)
        value = old.detach()[keep]
        addition = extra.get(name)
        if addition is not None: value = torch.cat([value, addition.to(device)], 0)
        parameter = nn.Parameter(value)
        state = optimizer.state.pop(old, {})
        new_state = {}
        for key, item in state.items():
            if torch.is_tensor(item) and item.shape == old.shape:
                selected = item[keep].clone()
                if addition is not None: selected = torch.cat([selected, torch.zeros_like(addition, device=device)], 0)
                new_state[key] = selected
            else: new_state[key] = item.clone() if torch.is_tensor(item) else item
        group['params'] = [parameter]; optimizer.state[parameter] = new_state
        setattr(model, name, parameter)
    for name in ['initial_xyz', 'point_ids', 'born_at']:
        value = getattr(model, name)[keep]
        if name in extra: value = torch.cat([value, extra[name].to(device)], 0)
        setattr(model, name, value)
    assert len(torch.unique(model.point_ids)) == len(model.point_ids)


@torch.no_grad()
def append_points(model, optimizer, xyz, rgb, ids, step, opacity=.05):
    device = model._xyz.device
    xyz = torch.as_tensor(xyz, dtype=model._xyz.dtype, device=device)
    rgb = torch.as_tensor(rgb, dtype=model._xyz.dtype, device=device)
    ids = torch.as_tensor(ids, dtype=torch.long, device=device)
    assert len(ids) and not torch.isin(ids, model.point_ids).any()
    rotations = torch.zeros(len(ids), 4, device=device); rotations[:, 0] = 1
    extra = {'_xyz': xyz, '_features_dc': ((rgb-.5)/.28209479177387814)[:, None],
        '_features_rest': torch.zeros(len(ids), 15, 3, device=device),
        '_opacity': torch.full((len(ids), 1), float(np.log(opacity/(1-opacity))), device=device),
        '_scaling': torch.full_like(xyz, float(np.log(model.spacing*.65))), '_rotation': rotations,
        'initial_xyz': xyz.clone(), 'point_ids': ids, 'born_at': torch.full_like(ids, step)}
    replace_rows(model, optimizer, torch.ones(len(model._xyz), dtype=torch.bool, device=device), extra)


@torch.no_grad()
def prune_points(model, optimizer, step, minimum_age=1000, opacity_threshold=.03):
    remove = (model.born_at > 0)&(step-model.born_at >= minimum_age)&(model.get_opacity[:, 0] < opacity_threshold)
    ids = model.point_ids[remove].detach().cpu().tolist()
    if ids: replace_rows(model, optimizer, ~remove)
    return ids
