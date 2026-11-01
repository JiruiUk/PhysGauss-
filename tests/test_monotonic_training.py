"""Legacy parity, differential physics, exact optimizer compatibility and export."""
import copy
import importlib.util
import json
from pathlib import Path
import torch
import pytest
from variants.monotonic_training.model import configure_clock,configure_from_env,time_coordinate

ROOT=Path(__file__).resolve().parents[1]


def load(relative):
    spec=importlib.util.spec_from_file_location('monotonic_test_'+relative.replace('/','_'),ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def nets():
    old=load('variants/integrity_tuning/code/motion_utils/pidg.py').VelocityNet(max_particles=32)
    new=load('variants/monotonic_training/code/motion_utils/pidg.py').VelocityNet(max_particles=32)
    new.load_state_dict(old.state_dict());return old,new


def test_flags_off_forward_gradient_and_rng_exact_parity():
    old,new=nets();x=torch.rand(12,4);ids=torch.arange(12)
    rng=torch.get_rng_state().clone();configure_clock(new,'legacy')
    assert torch.equal(rng,torch.get_rng_state())
    a=old(x,ids);b=new(x,ids)
    assert all(torch.equal(u,v) for u,v in zip(a,b))
    sum(v.square().sum() for v in a).backward();sum(v.square().sum() for v in b).backward()
    for p,q in zip(old.parameters(),new.parameters()):assert torch.equal(p.grad,q.grad)


def test_fixed_clock_monotonic_time_gradient_and_original_parameter_layout():
    old,new=nets();layout=[(k,tuple(p.shape)) for k,p in old.named_parameters()]
    configure_clock(new,'fixed_monotonic')
    assert layout==[(k,tuple(p.shape)) for k,p in new.named_parameters()]
    t=torch.linspace(0,1,1001)[:,None].requires_grad_();q=time_coordinate(new,t)
    assert (q.diff()>0).all();q.sum().backward();assert (t.grad>0).all()
    assert not any(p.requires_grad for m in [new.time_embed,new.time_proj] for p in m.parameters())


def test_original_optimizer_state_loads_without_dropping_groups_or_weights():
    old,new=nets();x=torch.rand(8,4);ids=torch.arange(8)
    a=torch.optim.Adam([{'params':[p],'lr':.001} for p in old.parameters()])
    sum(v.square().mean() for v in old(x,ids)).backward();a.step()
    state=copy.deepcopy(old.state_dict());optimizer=copy.deepcopy(a.state_dict())
    configure_clock(new,'fixed_monotonic');new.load_state_dict(state)
    b=torch.optim.Adam([{'params':[p],'lr':.001} for p in new.parameters()]);b.load_state_dict(optimizer)
    assert len(a.param_groups)==len(b.param_groups)
    for k,v in state.items():assert torch.equal(v,new.state_dict()[k])
    before={k:v.clone() for k,v in new.state_dict().items() if k.startswith(('time_embed.','time_proj.'))}
    b.zero_grad(set_to_none=True);new(x,ids)[0].square().mean().backward();b.step()
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_momentum_second_derivatives_and_stress_gradients(device):
    import os
    if device=='cuda' and os.environ.get('PIDG_MONOTONIC_GPU_TEST')!='1':pytest.skip('explicit GPU test')
    module=load('variants/monotonic_training/code/motion_utils/pidg.py')
    net=module.VelocityNet(max_particles=32).to(device);configure_clock(net,'fixed_monotonic')
    coords=(torch.rand(8,4,device=device)*.5).requires_grad_();ids=torch.arange(8,device=device)
    loss=module.cauchy_momentum_constraint(coords,lambda c:net(c,ids),torch.tensor(.2,device=device))
    loss.backward();assert torch.isfinite(loss)
    assert net.stress_head.weight.grad is not None and torch.isfinite(net.stress_head.weight.grad).all()
    assert net.velocity_head.weight.grad.abs().sum()>0
    assert all(p.grad is None for m in (net.time_embed,net.time_proj) for p in m.parameters())


def test_environment_flag_disabled_and_enabled(tmp_path,monkeypatch):
    _,new=nets();p=tmp_path/'config.json';monkeypatch.setenv('PIDG_METRIC_CONFIG',str(p))
    p.write_text(json.dumps({'enabled':False,'velocity_clock_mode':'fixed_monotonic'}))
    configure_from_env(new);assert not hasattr(new,'velocity_clock_mode')
    p.write_text(json.dumps({'enabled':True,'velocity_clock_mode':'fixed_monotonic'}))
    configure_from_env(new);assert new.velocity_clock_mode=='fixed_monotonic'
    with pytest.raises(ValueError):configure_clock(new,'unknown')
