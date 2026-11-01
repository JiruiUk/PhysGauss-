"""Verify isolation and the actual fixed-target regression before GPU execution."""
import importlib.util
from pathlib import Path
import torch
import pytest
from variants.frozen_velocity_probe.core import (
    freeze, state_hash, velocity_parameters, normalized_mse, alignment_metrics, disabled, partition_candidates)
from variants.kinematic_tuning.model import center_trajectory, trajectory_velocity


def original_net():
    path=Path(__file__).resolve().parents[1]/'variants/integrity_tuning/code/motion_utils/pidg.py'
    spec=importlib.util.spec_from_file_location('probe_test_pidg',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.VelocityNet(max_particles=64)


def test_disabled_no_rng_or_checkpoint_access():
    rng=torch.get_rng_state().clone()
    assert disabled({}) and disabled({'enabled':False})
    assert torch.equal(rng,torch.get_rng_state())


def test_frozen_rotating_and_translating_teacher_has_exact_derivative():
    t=torch.tensor([[0.],[.25],[1.]],dtype=torch.double)
    x=torch.tensor([[1.,0.,0.]],dtype=torch.double).expand(3,3)
    omega=torch.nn.Parameter(torch.tensor(1.,dtype=torch.double))
    teacher=torch.nn.ParameterList([omega]);freeze(teacher)
    before=state_hash(teacher)
    def trajectory(x,t):
        a=t*omega
        q=torch.cat([(a/2).cos(),a*0,a*0,(a/2).sin()],-1)
        return center_trajectory(x,torch.cat([q,t*t,t*0,t*0],-1))
    with torch.no_grad():v=trajectory_velocity(trajectory,x,t,1e-4)
    expected=torch.cat([-t.sin()+2*t,t.cos(),t*0],-1)
    torch.testing.assert_close(v,expected,atol=1e-7,rtol=1e-7)
    assert not v.requires_grad and omega.grad is None and state_hash(teacher)==before


def test_metrics_zero_prediction_is_100_percent_and_angles_undefined():
    target=torch.tensor([[1.,0.,0.],[0.,2.,0.]])
    metrics=alignment_metrics(target*0,target,.1)
    assert metrics['relative_rmse']==1 and metrics['direction_count']==0
    assert metrics['angle_median_deg'] is None and metrics['speed_ratio_median']==0
    exact=alignment_metrics(target,target,.1)
    assert exact['relative_rmse']==0 and exact['speed_ratio_median']==1
    assert exact['angle_median_deg']==0
    opposite=alignment_metrics(-target,target,.1)
    assert opposite['relative_rmse']==2 and opposite['angle_median_deg']==180


def test_original_velocity_network_receives_gradients_and_teacher_stays_frozen():
    torch.manual_seed(23)
    net=original_net();teacher=torch.nn.Linear(4,3);freeze(teacher)
    th=state_hash(teacher);sh=state_hash(net.stress_head)
    x=torch.rand(32,4)*1.5-.75;ids=torch.arange(32)
    with torch.no_grad():target=teacher(x)
    scale=target.square().mean().sqrt()
    params=velocity_parameters(net);opt=torch.optim.Adam(params,lr=.003)
    initial=normalized_mse(net(x,ids)[0],target,scale).item()
    first_grad={}
    for step in range(150):
        opt.zero_grad(set_to_none=True)
        loss=normalized_mse(net(x,ids)[0],target,scale);loss.backward()
        if not step:
            for name,p in net.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None and torch.isfinite(p.grad).all()
                    first_grad[name]=p.grad.abs().sum().item()
        opt.step()
    assert normalized_mse(net(x,ids)[0],target,scale).item()<initial*.05
    assert first_grad['velocity_head.weight']>0 and first_grad['planes']>0
    assert first_grad['index_embed.weight']>0
    assert state_hash(teacher)==th and state_hash(net.stress_head)==sh
    assert all(p.grad is None for p in teacher.parameters())


def test_rejects_attached_target():
    with pytest.raises(ValueError):
        normalized_mse(torch.ones(2,3),torch.ones(2,3,requires_grad=True),1.)


def test_particle_clone_families_are_disjoint_and_split_is_reproducible():
    ids=torch.arange(20).repeat_interleave(10)
    candidates=torch.arange(len(ids))
    rng=torch.get_rng_state().clone()
    rows=partition_candidates(candidates,ids,32,16,17)
    assert len(rows)==48 and len(rows.unique())==48
    assert not torch.isin(ids[rows[:32]],ids[rows[32:]]).any()
    assert torch.equal(rows,partition_candidates(candidates,ids,32,16,17))
    assert torch.equal(rng,torch.get_rng_state())
