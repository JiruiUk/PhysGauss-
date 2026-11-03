"""One canonical population and one unchanged original PIDG deformation field."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from ..prepare_data import ROOT, REPORT, RUNS, read, sha
from ..static_gaussian import StaticGaussian
from ..stage2.model import compose
from ..geometry_recovery.model import bounded_scale
from ..state_recovery.physics import CurrentWorldField
from ..train_static import PIPE


class UnionScene(nn.Module):
    def __init__(self, protocol):
        super().__init__()
        from scene_PIDG.deform_model import DeformModel
        report = REPORT/'S3_union'
        receipt = read(report/'data_receipt.json')
        initial = ROOT/receipt['initialization_file']
        assert sha(initial) == receipt['initialization_sha256']
        checkpoint = RUNS/'S1_static/checkpoint.pt'
        assert sha(checkpoint) == protocol['static_checkpoint_sha256']
        self.static = StaticGaussian(RUNS/read(REPORT/'S1_protocol.json')['initialization_file'])
        self.static.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
        self.static.requires_grad_(False)
        self.pools = nn.ModuleList([StaticGaussian(initial)])
        with np.load(initial) as data:
            self.register_buffer('centers', torch.tensor(data['center'][None], device='cuda'))
        deform = DeformModel(protocol['grid_args'], protocol['net_args'], reg_spatial_able=False, reg_temporal_able=False)
        for head in [deform.deform.gaussian_warp, deform.deform.gaussian_rotation, deform.deform.gaussian_scaling]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.grids = nn.ModuleList([deform.grid4d])
        self.networks = nn.ModuleList([deform.deform])
        self.deforms = [deform]
        # This population adapter does not mean a single physical body.
        self.physics = CurrentWorldField(1)

    @property
    def objects(self):
        """Compatibility view for the tested differential operator, not instances."""
        return self.pools

    def dynamic_at(self, t):
        from utils.general_utils import quaternion_to_matrix
        pool = self.pools[0]
        state = self.deforms[0].step(pool.get_xyz, pool.get_xyz.new_full((len(pool.get_xyz), 1), float(t)))
        xyz = (quaternion_to_matrix(state['d_xyz'][:, :4])@pool.get_xyz[..., None]).squeeze(-1)+state['d_xyz'][:, 4:]+self.centers[0]
        return SimpleNamespace(get_xyz=xyz, get_scaling=bounded_scale(pool._scaling, state['d_scaling'], pool.spacing, True),
            get_rotation=pool.get_rotation+state['d_rotation'], get_opacity=pool.get_opacity, get_features=pool.get_features,
            max_sh_degree=3, active_sh_degree=3)

    def render(self, renderer, cam, t):
        return renderer(cam, compose(self.static, [self.dynamic_at(t)]), PIPE, self.centers.new_zeros(3), 0., 0., 0.)

    @torch.no_grad()
    def constrain(self):
        pool = self.pools[0]
        pool._xyz.copy_(torch.maximum(torch.minimum(pool._xyz, pool.initial_xyz+.4), pool.initial_xyz-.4))
        pool._scaling.clamp_(np.log(pool.spacing*.15), np.log(.10))
        pool._opacity.clamp_(-12., 12.)

    def optimizer(self):
        rates = {'_xyz': .00016, '_features_dc': .0025, '_features_rest': .000125, '_opacity': .01, '_scaling': .002, '_rotation': .001}
        groups = [{'params': [getattr(self.pools[0], n)], 'lr': rate, 'name': 'dynamic/'+n} for n, rate in rates.items()]
        groups += [{'params': self.grids.parameters(), 'lr': .003, 'name': 'grids'},
                   {'params': self.networks.parameters(), 'lr': .0008, 'name': 'deform_network'},
                   {'params': self.physics.parameters(), 'lr': .001, 'name': 'PIDG_velocity_stress'}]
        return torch.optim.Adam(groups, eps=1e-15)
