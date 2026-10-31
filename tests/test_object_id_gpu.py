import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(os.environ.get('PIDG_OBJECT_GPU_TEST')!='1',reason='Explicit CUDA preflight')


def test_real_c1_flags_off_deformation_is_tensor_identical(tmp_path,monkeypatch):
    repo=ROOT/'variants/object_id/code';sys.path.insert(0,str(repo))
    config=tmp_path/'config.json';config.write_text(json.dumps({'enabled':True,'object_id_enabled':False}))
    monkeypatch.setenv('PIDG_METRIC_CONFIG',str(config))
    from scene_PIDG.deform_model import DeformModel
    path=ROOT/'variants/color_repair/code/scene_PIDG/deform_model.py'
    spec=importlib.util.spec_from_file_location('c1_reference_deformation',path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    settings=runpy.run_path(str(ROOT/'runs/pidg_color_repair/shared_prefix/rubber_duck/model_config.py'))
    args=[settings['grid_args'],settings['network_args']]
    scale=settings.get('scale_xyz',1.0)
    a=DeformModel(*args,scale_xyz=scale);b=original.DeformModel(*args,scale_xyz=scale)
    source=ROOT/'runs/pidg_color_repair/shared_prefix/rubber_duck/checkpoints/resume.pt'
    saved=torch.load(source,map_location='cpu',weights_only=False)
    grid,network=saved['grid4d'],saved['deform']
    for model in (a,b):model.grid4d.load_state_dict(grid);model.deform.load_state_dict(network)
    x=torch.linspace(-.7,.7,111,device='cuda').reshape(37,3);t=torch.full((37,1),.37,device='cuda')
    left,right=a.step(x,t),b.step(x,t)
    for key in ('d_xyz','d_rotation','d_scaling'):assert torch.equal(left[key],right[key])


def test_rendered_rgb_gradient_reaches_learned_identity(tmp_path,monkeypatch):
    path=ROOT/'tests/test_color_repair_gpu.py'
    spec=importlib.util.spec_from_file_location('c1_render_fixture',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    render,camera,pc,pipe,warp,_,_=helper.fixture(tmp_path,monkeypatch,True)
    from variants.object_id.model import ObjectMotion
    model=ObjectMotion(2).cuda();model.initialize(pc.get_xyz.detach())
    with torch.no_grad():model.motion[-1].bias.copy_(torch.linspace(-.2,.3,14,device='cuda'))
    changed,_=model(pc.get_xyz.detach(),torch.full((2,1),.4,device='cuda'),warp,.25)
    result=render(camera,pc,pipe,torch.zeros(3,device='cuda'),changed,0.,0.)
    (result['render']-.3).square().mean().backward()
    assert model.identity[-1].weight.grad.norm()>0
    assert torch.isfinite(model.identity[-1].weight.grad).all()
