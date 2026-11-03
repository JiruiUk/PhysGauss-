"""Frozen static bank + immutable canonical groups + original per-object PIDG."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from ..static_gaussian import StaticGaussian
from ..prepare_data import REPORT, RUNS, read, sha
from ..train_static import PIPE


def compose(static, objects):
    """Combine already transformed Gaussians before depth sorting and compositing."""
    names=('get_xyz','get_scaling','get_rotation','get_opacity','get_features')
    return SimpleNamespace(max_sh_degree=3,active_sh_degree=3,
        **{name:torch.cat([getattr(static,name)]+[getattr(o,name) for o in objects]) for name in names})


class ObjectScene(nn.Module):
    def __init__(self, protocol):
        super().__init__()
        from scene_PIDG.deform_model import DeformModel
        checkpoint=RUNS/'S1_static/checkpoint.pt'
        assert sha(checkpoint)==protocol['static_checkpoint_sha256']
        self.static=StaticGaussian(RUNS/read(REPORT/'S1_protocol.json')['initialization_file'])
        self.static.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
        self.static.requires_grad_(False)
        self.objects=nn.ModuleList();self.grids=nn.ModuleList();self.networks=nn.ModuleList();self.deforms=[]
        receipt=read(REPORT/'S2_data_receipt.json');centers=[];ids=[]
        for entry in receipt['objects']:
            path=RUNS/entry['file'];assert sha(path)==entry['sha256']
            obj=StaticGaussian(path);self.objects.append(obj)
            with np.load(path) as data:centers.append(data['center'])
            ids.extend([entry['object_id']]*len(obj.get_xyz))
            deform=DeformModel(protocol['grid_args'],protocol['net_args'],reg_spatial_able=False,reg_temporal_able=False)
            # Identity deformation at initialization; original forward architecture unchanged.
            for head in [deform.deform.gaussian_warp,deform.deform.gaussian_rotation,deform.deform.gaussian_scaling]:
                nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
            self.grids.append(deform.grid4d);self.networks.append(deform.deform);self.deforms.append(deform)
        self.register_buffer('centers',torch.tensor(np.asarray(centers),device='cuda'))
        self.register_buffer('object_ids',torch.tensor(ids,device='cuda',dtype=torch.long))

    def object_at(self,k,t):
        from utils.general_utils import quaternion_to_matrix
        obj=self.objects[k]
        state=self.deforms[k].step(obj.get_xyz,obj.get_xyz.new_full((len(obj.get_xyz),1),float(t)))
        q=quaternion_to_matrix(state['d_xyz'][:,:4])
        xyz=(q@obj.get_xyz[...,None]).squeeze(-1)+state['d_xyz'][:,4:]+self.centers[k]
        return SimpleNamespace(get_xyz=xyz,get_scaling=(obj.get_scaling+state['d_scaling']).clamp(1e-4,.15),
            get_rotation=obj.get_rotation+state['d_rotation'],get_opacity=obj.get_opacity,get_features=obj.get_features,
            max_sh_degree=3,active_sh_degree=3)

    def at(self,t):return compose(self.static,[self.object_at(k,t) for k in range(len(self.objects))])

    def render(self,renderer,cam,t):
        return renderer(cam,self.at(t),PIPE,self.centers.new_zeros(3),0.,0.,0.)

    @torch.no_grad()
    def constrain(self):
        for obj in self.objects:
            obj._xyz.copy_(torch.maximum(torch.minimum(obj._xyz,obj.initial_xyz+.4),obj.initial_xyz-.4))
            obj._scaling.clamp_(np.log(obj.spacing*.15),np.log(.10))
            obj._opacity.clamp_(-12.,12.)

    def optimizer(self):
        rates={'_xyz':.00016,'_features_dc':.0025,'_features_rest':.000125,'_opacity':.01,'_scaling':.002,'_rotation':.001}
        groups=[{'params':[getattr(obj,n)],'lr':rate,'name':f'object{k}/{n}'} for k,obj in enumerate(self.objects) for n,rate in rates.items()]
        groups.extend([{'params':self.grids.parameters(),'lr':.003,'name':'grids'},
                       {'params':self.networks.parameters(),'lr':.0008,'name':'deform_networks'}])
        return torch.optim.Adam(groups,eps=1e-15)
