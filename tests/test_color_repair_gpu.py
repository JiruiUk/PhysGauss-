"""CUDA parity, negative-channel gradients, viewing directions and full color state."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

pytestmark=pytest.mark.skipif(os.environ.get('PIDG_COLOR_REPAIR_GPU_TEST')!='1',reason='Explicit CUDA preflight')
ROOT=Path(__file__).resolve().parents[1]


def fixture(tmp_path,monkeypatch,positive=False):
    repo=ROOT/'variants/color_repair/code';sys.path.insert(0,str(repo))
    from gaussian_renderer import render
    from utils.graphics_utils import getProjectionMatrix
    from variants.color_repair.color import _read
    path=tmp_path/'render_config.json';path.write_text(json.dumps({'enabled':True,'positive_color':positive,'color_beta':5.}))
    monkeypatch.setenv('PIDG_METRIC_CONFIG',str(path));_read.cache_clear()
    camera=SimpleNamespace(image_height=64,image_width=64,FoVx=1.,FoVy=1.,
        world_view_transform=torch.eye(4,device='cuda'),full_proj_transform=getProjectionMatrix(.01,100,1.,1.).T.cuda(),
        camera_center=torch.zeros(3,device='cuda'))
    xyz=torch.tensor([[-.14,.05,2.],[.13,-.04,2.2]],device='cuda',requires_grad=True)
    coefficients=torch.zeros(2,16,3,device='cuda');coefficients[:,0,0]=(-.2-.5)/.28209479177387814
    coefficients.requires_grad_()
    pc=SimpleNamespace(get_xyz=xyz,get_opacity=torch.full((2,1),.4,device='cuda'),
        get_scaling=torch.full((2,3),.15,device='cuda'),get_rotation=torch.tensor([[1.,0.,0.,0.]]*2,device='cuda'),
        get_features=coefficients,active_sh_degree=0)
    pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
    shift=torch.tensor([[.17,.03,0.],[0.,0.,0.]],device='cuda',requires_grad=True)
    warp=torch.cat([pc.get_rotation,shift],-1)
    return render,camera,pc,pipe,warp,coefficients,shift


def test_flags_off_renderer_is_pixel_identical_to_original(tmp_path,monkeypatch):
    render,camera,pc,pipe,warp,_,_=fixture(tmp_path,monkeypatch,False)
    path=ROOT/'variants/coverage_tuning/code/gaussian_renderer/__init__.py'
    spec=importlib.util.spec_from_file_location('frozen_reference_renderer',path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    a=render(camera,pc,pipe,torch.zeros(3,device='cuda'),warp,0.,0.)
    b=original.render(camera,pc,pipe,torch.zeros(3,device='cuda'),warp,0.,0.)
    assert torch.equal(a['render'],b['render']) and torch.equal(a['alpha'],b['alpha'])


def test_softplus_restores_negative_red_gradient_and_matches_difference(tmp_path,monkeypatch):
    render,camera,pc,pipe,warp,coefficients,shift=fixture(tmp_path,monkeypatch,True)
    def objective():
        pkg=render(camera,pc,pipe,torch.zeros(3,device='cuda'),torch.cat([pc.get_rotation,shift],-1),0.,0.)
        return (pkg['render'][0]-.6).square().mean()
    loss=objective();loss.backward()
    assert coefficients.grad[:,0,0].abs().sum()>0 and torch.isfinite(coefficients.grad).all()
    assert shift.grad.abs().sum()>0 and torch.isfinite(shift.grad).all()
    analytic=coefficients.grad[0,0,0].item()
    with torch.no_grad():
        epsilon=.005
        coefficients[0,0,0]+=epsilon;plus=objective().item()
        coefficients[0,0,0]-=2*epsilon;minus=objective().item()
        coefficients[0,0,0]+=epsilon
    assert analytic==pytest.approx((plus-minus)/(2*epsilon),rel=.08,abs=1e-5)


def test_python_sh_uses_same_deformed_view_direction_as_cuda(tmp_path,monkeypatch):
    render,camera,pc,pipe,warp,coefficients,_=fixture(tmp_path,monkeypatch,False)
    from variants.color_repair.color import raw_sh_color
    with torch.no_grad():
        coefficients.zero_();coefficients[:,1:,:]=torch.linspace(-.04,.04,90,device='cuda').reshape(2,15,3)
    pc.active_sh_degree=3
    a=render(camera,pc,pipe,torch.zeros(3,device='cuda'),warp,0.,0.)
    color=raw_sh_color(pc,a['means3D_deform'],camera).clamp_min(0)
    b=render(camera,pc,pipe,torch.zeros(3,device='cuda'),warp,0.,0.,override_color=color)
    assert torch.allclose(a['render'],b['render'],atol=2e-6,rtol=2e-5)


def test_color_evidence_checkpoint_roundtrip_and_checksum(tmp_path):
    from variants.color_repair.runtime import Tuning
    from variants.color_repair.evidence import ColorEvidence
    from variants.coverage_tuning.density import CoverageAccumulator
    from variants.metric_tuning.runtime import sha256
    path=tmp_path/'config.json';path.write_text(json.dumps({'enabled':True,'coverage_density':True}))
    checkpoint=tmp_path/'resume.pt';checkpoint.write_bytes(b'isolated full-state placeholder')
    a=Tuning(path,tmp_path);a.target_ids=['0002'];a.coverage=CoverageAccumulator(4,1,'cuda')
    a.color_evidence=ColorEvidence(4,'cuda')
    ids=torch.tensor([[[1]]],device='cuda');deficit=torch.ones(1,1,1,device='cuda');rgb=torch.full((3,1,1),.4,device='cuda')
    a.coverage.observe(0,ids,deficit);a.color_evidence.observe(ids,deficit,rgb)
    a.save(checkpoint,{'sha256':sha256(checkpoint),'iteration':47})
    b=Tuning(path,tmp_path);b.target_ids=a.target_ids;b.restore(checkpoint,47)
    assert b.color_evidence.first.is_cuda and torch.equal(a.color_evidence.first,b.color_evidence.first)
    evidence=checkpoint.with_suffix('.color.pt');evidence.write_bytes(evidence.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'):b.restore(checkpoint,47)
