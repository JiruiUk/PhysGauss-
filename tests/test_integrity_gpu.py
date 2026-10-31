import importlib.util
import os
from pathlib import Path
import pytest
import torch
from variants.integrity_tuning.model import coverage_objective
pytestmark=pytest.mark.skipif(os.environ.get('PIDG_INTEGRITY_GPU_TEST')!='1',reason='Explicit queued CUDA preflight')


def test_real_raster_alpha_loss_backpropagates(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parent/'test_color_repair_gpu.py'
    spec=importlib.util.spec_from_file_location('integrity_render_fixture',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    render,camera,pc,pipe,warp,_,_=helper.fixture(tmp_path,monkeypatch,True)
    pc.get_opacity.requires_grad_()
    package=render(camera,pc,pipe,torch.zeros(3,device='cuda'),warp,0.,0.)
    loss=coverage_objective(package['alpha'],torch.ones_like(package['alpha']))
    loss.backward()
    assert torch.isfinite(loss) and pc.get_opacity.grad.abs().sum()>0
    assert torch.isfinite(pc.get_xyz.grad).all()
