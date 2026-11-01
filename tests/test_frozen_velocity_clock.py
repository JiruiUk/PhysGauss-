import importlib.util
from pathlib import Path
import torch
import pytest
from variants.frozen_velocity_probe.clock import apply_clock
from variants.frozen_velocity_probe.core import state_hash


def net():
    spec=importlib.util.spec_from_file_location('clock_test_pidg',Path(__file__).resolve().parents[1]/'variants/integrity_tuning/code/motion_utils/pidg.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.VelocityNet(max_particles=16)


def test_legacy_clock_is_noop_and_consumes_no_rng():
    model=net();before=state_hash(model);rng=torch.get_rng_state().clone()
    assert apply_clock(model) is model
    assert state_hash(model)==before and torch.equal(rng,torch.get_rng_state())


def test_fixed_clock_is_monotonic_finite_and_freezes_only_clock():
    model=net();head=state_hash(model.velocity_head);planes=model.planes.detach().clone()
    apply_clock(model,'fixed_monotonic')
    t=torch.linspace(0,1,1001)[:,None].requires_grad_()
    q=torch.tanh(model.time_proj(model.time_embed(t)))
    torch.testing.assert_close(q,torch.tanh(2*t-1))
    assert torch.isfinite(q).all() and (q.diff(dim=0)>0).all()
    assert not any(p.requires_grad for p in model.time_proj.parameters())
    assert model.velocity_head.weight.requires_grad and model.planes.requires_grad
    assert state_hash(model.velocity_head)==head and torch.equal(model.planes,planes)
    q.sum().backward();assert torch.isfinite(t.grad).all() and (t.grad>0).all()


def test_unknown_clock_is_rejected():
    with pytest.raises(ValueError):apply_clock(net(),'unknown')
