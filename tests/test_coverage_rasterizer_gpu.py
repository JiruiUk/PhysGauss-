"""Actual alpha backward, not merely gradients with respect to an alpha leaf.

Run explicitly with PIDG_COVERAGE_GPU_TEST=1 and the enabled CUDA environment.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('PIDG_COVERAGE_GPU_TEST') != '1',reason='Explicit CUDA rasterizer check')


def test_cuda_alpha_gradients_reach_gaussian_and_warp_and_match_finite_difference():
    repo = Path(__file__).resolve().parents[1]/'variants/coverage_tuning/code'
    sys.path.insert(0,str(repo))
    from gaussian_renderer import render
    from utils.graphics_utils import getProjectionMatrix
    from variants.coverage_tuning.model import mask_loss
    xyz = torch.tensor([[-.14,.05,2.],[.13,-.04,2.2]],device='cuda',requires_grad=True)
    logits = torch.tensor([[-.5],[-.2]],device='cuda',requires_grad=True)
    scale = torch.tensor([[.13,.17,.15],[.19,.12,.17]],device='cuda',requires_grad=True)
    shift = torch.zeros(2,3,device='cuda',requires_grad=True)
    quat = torch.tensor([[1.,0.,0.,0.]]*2,device='cuda')
    camera = SimpleNamespace(image_height=64,image_width=64,FoVx=1.,FoVy=1.,
        world_view_transform=torch.eye(4,device='cuda'),
        full_proj_transform=getProjectionMatrix(.01,100,1.,1.).T.cuda(),camera_center=torch.zeros(3,device='cuda'))
    pipe = SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
    gt = torch.zeros(1,64,64,device='cuda'); gt[:,25:40,23:41] = 1
    def objective():
        pc = SimpleNamespace(get_xyz=xyz,get_opacity=logits.sigmoid(),get_scaling=scale,
            get_rotation=quat,get_features=torch.zeros(2,16,3,device='cuda'),active_sh_degree=0)
        return mask_loss(render(camera,pc,pipe,torch.zeros(3,device='cuda'),
            torch.cat([quat,shift],-1),0.,0.)['alpha'],gt)
    loss = objective(); loss.backward()
    for value in (xyz,logits,scale,shift):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    analytic = logits.grad[0,0].item()
    with torch.no_grad():
        epsilon = .01
        logits[0,0] += epsilon; plus = objective().item()
        logits[0,0] -= 2*epsilon; minus = objective().item()
        logits[0,0] += epsilon
    numerical = (plus-minus)/(2*epsilon)
    assert analytic == pytest.approx(numerical,rel=.08,abs=3e-5)


def test_nonempty_coverage_window_is_checksum_bound_and_restored_on_gpu(tmp_path):
    import json
    from variants.coverage_tuning.runtime import Tuning
    from variants.coverage_tuning.density import CoverageAccumulator
    from variants.metric_tuning.runtime import sha256
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'enabled':True,'coverage_density':True}))
    checkpoint = tmp_path/'resume.pt'; checkpoint.write_bytes(b'local-test-full-state-placeholder')
    a = Tuning(config,tmp_path); a.target_ids = ['0002','0003']
    a.coverage = CoverageAccumulator(4,2,'cuda')
    a.coverage.observe(0,torch.tensor([[[2]]],device='cuda'),torch.ones(1,1,1,device='cuda'))
    a.save(checkpoint,{'sha256':sha256(checkpoint),'iteration':47})
    b = Tuning(config,tmp_path); b.target_ids = a.target_ids
    b.restore(checkpoint,47)
    assert b.coverage.score.is_cuda and torch.equal(a.coverage.score,b.coverage.score)
    assert torch.equal(a.coverage.votes,b.coverage.votes) and b.coverage.votes.sum() == 1
    path = checkpoint.with_suffix('.coverage.pt'); path.write_bytes(path.read_bytes()+b'altered')
    with pytest.raises(ValueError,match='checksum'): b.restore(checkpoint,47)
