"""Verify signed centroid gradients through the real CUDA alpha renderer."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('PIDG_CENTROID_GPU_TEST') != '1', reason='Explicit GPU preflight')


def test_centroid_gradients_reach_actual_gaussian_geometry_and_match_difference():
    repo = Path(__file__).resolve().parents[1] / 'variants/centroid_tuning/code'
    sys.path.insert(0, str(repo))
    from gaussian_renderer import render
    from utils.graphics_utils import getProjectionMatrix
    from variants.centroid_tuning.model import centroid_loss
    xyz = torch.tensor([[-.14,.05,2.],[.13,-.04,2.2]], device='cuda', requires_grad=True)
    logits = torch.tensor([[-.5],[-.2]], device='cuda', requires_grad=True)
    scale = torch.tensor([[.13,.17,.15],[.19,.12,.17]], device='cuda', requires_grad=True)
    shift = torch.zeros(2,3, device='cuda', requires_grad=True)
    quat = torch.tensor([[1.,0.,0.,0.]]*2, device='cuda')
    colors = torch.zeros(2,16,3, device='cuda', requires_grad=True)
    camera = SimpleNamespace(image_height=64,image_width=64,FoVx=1.,FoVy=1.,
        world_view_transform=torch.eye(4,device='cuda'),
        full_proj_transform=getProjectionMatrix(.01,100,1.,1.).T.cuda(),camera_center=torch.zeros(3,device='cuda'))
    pipe = SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
    def objective():
        pc = SimpleNamespace(get_xyz=xyz,get_opacity=logits.sigmoid(),get_scaling=scale,
            get_rotation=quat,get_features=colors,active_sh_degree=0)
        alpha = render(camera,pc,pipe,torch.zeros(3,device='cuda'),torch.cat([quat,shift],-1),0.,0.)['alpha']
        return centroid_loss(alpha, torch.tensor([.7,.3],device='cuda'))
    loss = objective(); loss.backward()
    for value in (xyz,logits,scale,shift):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    assert colors.grad is None or torch.count_nonzero(colors.grad)==0
    analytic = shift.grad[0,0].item()
    with torch.no_grad():
        epsilon = .001
        shift[0,0] += epsilon; plus = objective().item()
        shift[0,0] -= 2*epsilon; minus = objective().item()
        shift[0,0] += epsilon
    numerical = (plus-minus)/(2*epsilon)
    assert analytic == pytest.approx(numerical,rel=.1,abs=3e-5)
