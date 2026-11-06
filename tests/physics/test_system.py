import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from physics.system import MechanicsSystem
from models.continuous_motion import ContinuousSE3Motion

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from upgrade_configs import stage_configs


class ToyGrid(nn.Module):
    bound=1.
    def forward(self,xyzt): return xyzt[:,:3],xyzt[:,3:]


class ToyMotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficient=nn.Parameter(torch.tensor(.01,dtype=torch.float64))
    def forward(self,X,t):
        translation=self.coefficient*torch.sin(X)*t.square()
        q=torch.cat([torch.ones_like(t),t.expand(-1,3)*0],-1)
        motion=torch.cat([q,translation],-1)
        return motion,X*0,X*0


class ToyMaterial(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Linear(4,9).double()
    def forward(self,x,indices=None):
        output=torch.tanh(self.net(x))
        return output[:,:3],output[:,3:]


@pytest.mark.parametrize("phase",range(2,16))
def test_all_enabled_stages_backward_and_diagnostics(phase,tmp_path,monkeypatch):
    path=tmp_path/'config.json';path.write_text(json.dumps(stage_configs()[phase]))
    monkeypatch.setenv('PIDG_UPGRADE_CONFIG',str(path))
    monkeypatch.setenv('PIDG_UPGRADE_SMOKE','1')
    torch.manual_seed(7)
    system=MechanicsSystem().double()
    motion=ToyMotion()
    if phase>=14:
        motion.motion_basis=ContinuousSE3Motion().double()
        motion.motion_basis.field.bias.data.normal_(std=.001)
    deform=SimpleNamespace(grid4d=ToyGrid(),deform=motion,scale_xyz=torch.ones(3,dtype=torch.float64))
    g=SimpleNamespace(get_xyz=torch.randn(24,3,dtype=torch.float64)*.2,
                      get_features=torch.randn(24,1,3,dtype=torch.float64),
                      particle_indices=torch.arange(24),velocity_net=ToyMaterial())
    loss,diag=system(g,deform,torch.tensor([.2357],dtype=torch.float64),100,tmp_path)
    loss.backward()
    assert torch.isfinite(loss) and diag['loss_kin'] is not None
    for p in list(system.parameters())+list(motion.parameters())+list(g.velocity_net.parameters()):
        if p.grad is not None: assert torch.isfinite(p.grad).all()
    system.write_diagnostics(tmp_path,100,diag,torch.tensor(.1))
    record=json.loads((tmp_path/'physics.jsonl').read_text())
    assert record['mean_speed'] is not None
    if phase>=4: assert record['min_detF']>0
    if phase>=8: assert record['loss_momentum'] is not None
    if phase==15: assert 'loss_rollout' in record
