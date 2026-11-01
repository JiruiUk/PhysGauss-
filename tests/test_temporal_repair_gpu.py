"""Real CUDA rasterizer checks for opt-in covariance normalization."""
import importlib.util
import json
import os
from pathlib import Path
import pytest
import torch
pytestmark=pytest.mark.skipif(os.environ.get('PIDG_TEMPORAL_GPU_TEST')!='1',reason='Explicit temporal repair CUDA preflight')


def helper():
    path=Path(__file__).parent/'test_color_repair_gpu.py'
    spec=importlib.util.spec_from_file_location('temporal_render_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def renderer():
    path=Path(__file__).resolve().parents[1]/'variants/temporal_repair/code/gaussian_renderer/__init__.py'
    spec=importlib.util.spec_from_file_location('repaired_covariance_renderer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.render


def test_flags_off_matches_frozen_h3_pixels_exactly(tmp_path,monkeypatch):
    original,view,pc,pipe,warp,_,_=helper().fixture(tmp_path,monkeypatch,True)
    a=original(view,pc,pipe,torch.zeros(3,device='cuda'),warp,0.,0.)
    b=renderer()(view,pc,pipe,torch.zeros(3,device='cuda'),warp,0.,0.)
    for k in ['render','alpha','radii','depth']:assert torch.equal(a[k],b[k])


def test_normalized_covariance_backpropagates_to_rotation_offset(tmp_path,monkeypatch):
    _,view,pc,pipe,warp,_,_=helper().fixture(tmp_path,monkeypatch,True)
    from variants.color_repair.color import _read
    path=Path(os.environ['PIDG_METRIC_CONFIG']);c=json.loads(path.read_text());c['integrity_unit_rotation']=True
    path.write_text(json.dumps(c));_read.cache_clear()
    pc.get_scaling[:,1]*=.5
    offset=torch.tensor([[.3,.1,.2,.05]]*2,device='cuda',requires_grad=True)
    package=renderer()(view,pc,pipe,torch.zeros(3,device='cuda'),warp,offset,0.)
    package['render'].square().sum().backward()
    assert offset.grad is not None and torch.isfinite(offset.grad).all() and offset.grad.abs().sum()>0
