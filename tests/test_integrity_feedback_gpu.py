import importlib.util
import os
from pathlib import Path
import torch
import pytest
from variants.integrity_feedback.model import foreground_tail_loss
pytestmark=pytest.mark.skipif(os.environ.get('PIDG_FEEDBACK_GPU_TEST')!='1',reason='Queued GPU preflight')


def test_tail_reconstruction_reaches_actual_gaussian_color(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parent/'test_color_repair_gpu.py'
    spec=importlib.util.spec_from_file_location('feedback_render_fixture',path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    render,camera,pc,pipe,warp,coefficients,_=fixture.fixture(tmp_path,monkeypatch,True)
    package=render(camera,pc,pipe,torch.zeros(3,device='cuda'),warp,0.,0.)
    mask=(package['alpha'].detach()>.05).float()
    loss=foreground_tail_loss(package['render'],torch.full_like(package['render'],.8),mask)
    loss.backward()
    assert coefficients.grad[:,0,0].abs().sum()>0 and torch.isfinite(coefficients.grad).all()
