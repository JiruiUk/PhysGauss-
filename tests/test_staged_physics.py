import pytest
import torch
from variants.staged_reconstruction.stage3.physics import material_acceleration,qualified_huber
from variants.coherent_momentum.operator import divergence


def test_material_time_derivative_uses_real_normalized_gap_without_double_convection():
    X=torch.tensor([[2.,3.,4.]],dtype=torch.float64)
    a,t=material_acceleration(lambda s:X+torch.tensor([[s*s,2*s,0.]],dtype=torch.float64),.4,1/199)
    assert t==.4 and torch.allclose(a,torch.tensor([[.8,2.,0.]],dtype=torch.float64),atol=1e-10)


def test_interval_boundary_is_recorded_and_constant_material_velocity_has_zero_acceleration():
    a,t=material_acceleration(lambda s:torch.ones(4,3,dtype=torch.float64),0.,1/199)
    assert t==1/199 and torch.equal(a,torch.zeros_like(a))


def test_invalid_neighborhoods_cannot_pass_as_zero_momentum():
    with pytest.raises(ValueError,match='nondegenerate'):
        qualified_huber(torch.zeros(4,3),torch.zeros(4,dtype=torch.bool))


def test_manufactured_stress_divergence_and_velocity_gradient_backpropagate():
    torch.manual_seed(33);x=torch.randn(80,3,dtype=torch.float64)*.2
    scale=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    stress=torch.stack([scale*x[:,0],3*x[:,1],4*x[:,2],x[:,0]*0,x[:,0]*0,x[:,0]*0],-1)
    velocity=x*0
    anchors=torch.tensor([0,1,2,3]);neighbor=torch.stack([torch.tensor([j for j in range(80) if j!=int(i)]) for i in anchors])
    d,valid,_=divergence(x,velocity,stress,torch.tensor(1.),anchors,neighbor,ridge=1e-10)
    assert valid.all() and torch.allclose(d,torch.tensor([2.,3.,4.]).double().expand(4,3),atol=1e-6)
    d.square().sum().backward();assert scale.grad.isfinite() and scale.grad>0


def test_unreviewed_S2_cannot_start_physics_training(tmp_path,monkeypatch):
    from variants.staged_reconstruction.stage3 import train
    monkeypatch.setattr(train,'REC_REPORT',tmp_path)
    with pytest.raises(RuntimeError,match='not been reviewed'):
        train.parent_admission()


def test_stage3_disabled_entry_does_not_initialize_renderer():
    import subprocess,sys
    p=subprocess.run([sys.executable,'-m','variants.staged_reconstruction.stage3.train'],capture_output=True,text=True)
    assert p.returncode!=0 and 'opt-in' in p.stderr and 'ninja' not in p.stdout
