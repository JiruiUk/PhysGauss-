"""Native PIDG temporal field shared by all unlabeled dynamic Gaussian rows."""
import math
from types import SimpleNamespace
import torch
from torch import nn
from ..prepare_data import ROOT, read, sha
from ..adaptive_static.model import AdaptiveGaussian
from ..train_static import optimizer_for, PIPE
from ..stage2.model import compose
from ..geometry_recovery.model import bounded_scale
from .core import open_static_appearance, full_rgb_loss, restore_rng
from .io import BASE


class JointScene(nn.Module):
    def __init__(self, protocol, case, revisable=False):
        super().__init__()
        from scene_PIDG.deform_model import DeformModel
        parent = protocol['parents'][case]
        assert sha(ROOT/parent['checkpoint']) == parent['sha256']
        state = torch.load(ROOT/parent['checkpoint'], map_location='cpu', weights_only=False)
        assert state['iteration'] == 6000 and state['nuisance'] is None
        self.static = AdaptiveGaussian(ROOT/parent['initialization'])
        self.static.load_payload(state['model']); self.static.requires_grad_(False)
        open_static_appearance(self.static, revisable)
        self.dynamic = AdaptiveGaussian(ROOT/parent['initialization'])
        self.dynamic.load_payload(state['model'])
        assert len(self.static._xyz) == len(self.dynamic._xyz) == 4000
        self.register_buffer('center', self.dynamic._xyz.detach().mean(0))
        with torch.no_grad():
            self.dynamic._xyz.sub_(self.center)
            self.dynamic.initial_xyz.copy_(self.dynamic._xyz)
            opacity = protocol['initial_dynamic_opacity']
            self.dynamic._opacity.fill_(math.log(opacity/(1-opacity)))
        torch.manual_seed(protocol['seed'])
        deform = DeformModel(protocol['grid_args'], protocol['net_args'],
            reg_spatial_able=False, reg_temporal_able=False)
        for head in [deform.deform.gaussian_warp, deform.deform.gaussian_rotation, deform.deform.gaussian_scaling]:
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        self.grid = deform.grid4d; self.network = deform.deform; self.deform = deform
        self.parent = state

    def at(self, t):
        from utils.general_utils import quaternion_to_matrix
        g = self.dynamic
        delta = self.deform.step(g.get_xyz, g.get_xyz.new_full((len(g.get_xyz), 1), float(t)))
        xyz = (quaternion_to_matrix(delta['d_xyz'][:, :4])@g.get_xyz[..., None]).squeeze(-1)
        return SimpleNamespace(get_xyz=xyz+delta['d_xyz'][:, 4:]+self.center,
            get_scaling=bounded_scale(g._scaling, delta['d_scaling'], g.spacing, True),
            get_rotation=g.get_rotation+delta['d_rotation'], get_opacity=g.get_opacity,
            get_features=g.get_features, max_sh_degree=3, active_sh_degree=3)

    def render(self, renderer, camera, t):
        return renderer(camera, compose(self.static, [self.at(t)]), PIPE,
            self.center.new_zeros(3), 0., 0., 0.)

    def optimizers(self, protocol):
        static = optimizer_for(self.static); static.load_state_dict(self.parent['optimizer'])
        rates = protocol['dynamic_rates']
        groups = [{'params': [getattr(self.dynamic, name)], 'lr': rates[name], 'name': 'dynamic/'+name}
            for name in ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation']]
        groups += [{'params': self.grid.parameters(), 'lr': rates['grids'], 'name': 'grids'},
            {'params': self.network.parameters(), 'lr': rates['deform_network'], 'name': 'deform_network'}]
        return static, torch.optim.Adam(groups, eps=protocol['Adam_eps'])

    @torch.no_grad()
    def constrain(self):
        g = self.dynamic
        g._xyz.copy_(torch.maximum(torch.minimum(g._xyz, g.initial_xyz+.4), g.initial_xyz-.4))
        g._scaling.clamp_(math.log(g.spacing*.15), math.log(.10)); g._opacity.clamp_(-12, 12)


def loss(package, target):
    from utils.loss_utils import ssim
    return full_rgb_loss(package['render'], target, ssim)
