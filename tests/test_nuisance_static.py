import copy
import random
import numpy as np
import torch
import pytest
from variants.staged_reconstruction.nuisance_static.core import (compose,NuisancePlane,optimizer_for_nuisance,support_penalty)


def model(seed=5):
    k = np.array([[100.,0,31.5],[0,100,17.5],[0,0,1.]])
    image = np.ones((36,64,3),np.float32)*.4
    return NuisancePlane(np.eye(4),k,image,count=16,seed=seed,device='cpu'),k


def test_flag_off_returns_original_tensor_object_without_using_nuisance():
    original = torch.rand(3,4,7)
    assert compose(original) is original
    assert compose(original,torch.full_like(original,float('nan')),torch.zeros(1,4,7)) is original


def test_premultiplied_composition_and_gradients():
    shared = torch.full((3,2,2),.6,requires_grad=True)
    rgb = torch.full((3,2,2),.2,requires_grad=True); alpha = torch.full((1,2,2),.25,requires_grad=True)
    value = compose(shared,rgb,alpha,enabled=True)
    assert torch.allclose(value,torch.full_like(value,.65))
    value.sum().backward()
    assert shared.grad.isfinite().all() and rgb.grad.isfinite().all() and alpha.grad.isfinite().all()
    assert alpha.grad.abs().sum() > 0


def test_camera_plane_projects_to_initialized_pixel_domain():
    plane,k = model(); points = plane.get_xyz.detach().numpy()
    assert np.allclose(points[:,2],.3)
    projected = points@k.T; uv = projected[:,:2]/projected[:,2:]
    expected = plane._uv.detach().numpy()*np.array([32,18])+np.array([31.5,17.5])
    assert np.allclose(uv,expected,atol=3e-6)
    assert np.allclose(torch.linalg.vector_norm(plane.get_rotation,dim=1),1)


def test_constructor_does_not_consume_global_RNG():
    a,b,c = random.getstate(),np.random.get_state(),torch.get_rng_state()
    model()
    assert a == random.getstate() and np.array_equal(b[1],np.random.get_state()[1])
    assert torch.equal(c,torch.get_rng_state())


def test_only_selected_view_receives_updates_and_full_Adam_recovers():
    bank = [model(i)[0] for i in range(3)]; opts = [optimizer_for_nuisance(m) for m in bank]
    untouched = [copy.deepcopy(m.state_dict()) for m in bank]
    loss = sum(p.square().sum() for p in bank[1].parameters()); loss.backward(); opts[1].step()
    assert all(p.grad is None for i in [0,2] for p in bank[i].parameters())
    assert all(torch.equal(v,bank[i].state_dict()[n]) for i in [0,2] for n,v in untouched[i].items())
    restored,_ = model(1); restored.load_state_dict(bank[1].state_dict()); opt = optimizer_for_nuisance(restored)
    opt.load_state_dict(copy.deepcopy(opts[1].state_dict()))
    assert [int(v['step']) for v in opt.state.values()] == [1]*4
    opt.zero_grad(); sum(p.square().sum() for p in restored.parameters()).backward(); opt.step()
    assert [int(v['step']) for v in opt.state.values()] == [2]*4


def test_unknown_background_has_no_static_alpha_one_pressure():
    alpha = torch.full((1,3,5),.2,requires_grad=True); reliable = torch.zeros(3,5,dtype=torch.bool)
    value = support_penalty(alpha,reliable,None); value.backward()
    assert value == 0 and not alpha.grad.any()
    alpha.grad = None; reliable[1,2] = True
    support_penalty(alpha,reliable,None).backward()
    assert torch.count_nonzero(alpha.grad) == 1 and alpha.grad[0,1,2] < 0


def test_derived_world_parameters_all_receive_finite_gradients():
    plane,_ = model()
    loss = plane.get_xyz.square().sum()+plane.get_scaling.sum()+plane.get_rgb.sum()+plane.get_opacity.sum()
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0 for p in plane.parameters())


@pytest.mark.parametrize('entry',['prepare','train','native_check'])
def test_entries_are_opt_in(entry):
    import subprocess,sys
    p = subprocess.run([sys.executable,'-m','variants.staged_reconstruction.nuisance_static.'+entry],capture_output=True,text=True)
    assert p.returncode != 0 and 'opt-in' in p.stderr
