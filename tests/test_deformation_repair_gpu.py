import importlib.util
import os
from pathlib import Path
import sys
import pytest
import torch
pytestmark=pytest.mark.skipif(os.environ.get('PIDG_DEFORMATION_REPAIR_GPU_TEST')!='1',reason='Explicit model-repair CUDA preflight')


def test_covariance_matches_actual_cuda_nonunit_rotation_and_signed_scale(tmp_path,monkeypatch):
    path=Path(__file__).parent/'test_color_repair_gpu.py';spec=importlib.util.spec_from_file_location('deformation_fixture',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    render,view,pc,pipe,warp,_,_=helper.fixture(tmp_path,monkeypatch,True)
    from variants.deformation_repair.model import rendered_covariance,packed_covariance
    offset=torch.tensor([[.3,.1,.2,-.05]]*2,device='cuda');ds=torch.tensor([[-.2,.1,.0],[.03,-.3,.01]],device='cuda')
    a=render(view,pc,pipe,torch.zeros(3,device='cuda'),warp,offset,ds)
    cov=packed_covariance(rendered_covariance(pc.get_scaling+ds,pc.get_rotation+offset))
    pc.get_covariance=lambda modifier:cov;pipe.compute_cov3D_python=True
    b=render(view,pc,pipe,torch.zeros(3,device='cuda'),warp,offset,ds)
    assert torch.allclose(a['render'],b['render'],atol=2e-6,rtol=3e-5)
    assert torch.allclose(a['alpha'],b['alpha'],atol=2e-6,rtol=3e-5)


def test_covariance_regularizer_reaches_real_hash_and_deformation_heads():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'variants/integrity_tuning/code'))
    from scene_PIDG.deform_model import DeformModel
    from variants.deformation_repair.model import rendered_covariance,covariance_curvature
    torch.manual_seed(7)
    grid={'canonical_num_levels':2,'canonical_level_dim':2,'canonical_desired_resolution':32,'canonical_log2_hashmap_size':8,
        'deform_num_levels':2,'deform_level_dim':2,'deform_desired_resolution':[16,16,16],'deform_log2_hashmap_size':8}
    deform=DeformModel(grid,{'width':16,'depth':0},reg_spatial_able=False,reg_temporal_able=False)
    x=torch.rand(32,3,device='cuda')*.3;rotation=x.new_tensor([1.,0,0,0]).expand(32,-1);scale=torch.full_like(x,.01);cov=[]
    for t in [.2,.5,.8]:
        p=deform.step(x,x.new_full((len(x),1),t));cov.append(rendered_covariance(scale+p['d_scaling'],rotation+p['d_rotation']))
    loss,_=covariance_curvature(*cov,shape_tolerance=0.,size_tolerance=0.);loss.backward()
    for parameters in [deform.grid4d.parameters(),deform.deform.gaussian_rotation.parameters(),deform.deform.gaussian_scaling.parameters()]:
        g=[p.grad for p in parameters if p.grad is not None]
        assert g and all(torch.isfinite(v).all() for v in g) and sum(v.abs().sum().item() for v in g)>0
