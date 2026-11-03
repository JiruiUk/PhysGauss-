import random
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest
import torch
from variants.staged_reconstruction.joint_revocation_audit.core import (
    audit_scope, state_hash, rng_state, rng_equal, preserve_rng, gradient_stats)


def scene():
    result = torch.nn.Module(); result.static = torch.nn.Module()
    for name in ['_xyz', '_opacity', '_features_dc', '_features_rest']:
        result.static.register_parameter(name, torch.nn.Parameter(torch.ones(2), requires_grad=False))
    result.dynamic = torch.nn.Parameter(torch.ones(2))
    result.register_buffer('buffer', torch.tensor([3.]))
    return result


def test_default_off_same_object_and_flags():
    model = scene(); before = state_hash(model)
    with audit_scope(model) as result:
        assert result is model and not model.static._opacity.requires_grad
    assert state_hash(model) == before


def test_enabled_opacity_appearance_only_with_dynamic_preserved():
    model = scene()
    with audit_scope(model, True):
        assert model.static._opacity.requires_grad and model.static._features_dc.requires_grad
        assert model.static._features_rest.requires_grad and not model.static._xyz.requires_grad
        assert model.dynamic.requires_grad
    assert not model.static._opacity.requires_grad


def test_gradients_flow_without_leaf_grad_accumulation():
    model = scene()
    with audit_scope(model, True):
        params = [model.static._opacity, model.static._features_dc, model.dynamic]
        loss = (params[0]*params[1]*params[2]).sum()
        gradients = torch.autograd.grad(loss, params)
        assert all(g.isfinite().all() and g.abs().sum() > 0 for g in gradients)
        assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('fail', [False, True])
def test_restores_values_flags_buffers_and_existing_grad_even_on_failure(fail):
    model = scene(); model.static._xyz.requires_grad_(True)
    model.dynamic.grad = torch.tensor([7., 8.]); original_grad = model.dynamic.grad
    before = state_hash(model); flags = [p.requires_grad for p in model.parameters()]
    try:
        with audit_scope(model, True):
            with torch.no_grad():
                model.dynamic.add_(1); model.buffer.add_(4); original_grad.add_(5)
            model.dynamic.grad = None
            if fail:
                raise ValueError('injected')
    except ValueError:
        pass
    assert state_hash(model) == before and [p.requires_grad for p in model.parameters()] == flags
    assert model.dynamic.grad is original_grad and torch.equal(original_grad, torch.tensor([7., 8.]))


def test_rng_preserved_with_random_draws_including_exception():
    model = scene(); before = rng_state()
    with pytest.raises(ValueError):
        with audit_scope(model, True):
            random.random(); np.random.random(); torch.rand(3)
            raise ValueError('injected')
    assert rng_equal(before, rng_state())


def test_outer_rng_guard_protects_model_construction():
    before = rng_state()
    with preserve_rng():
        torch.nn.Linear(3, 2); random.random(); np.random.random()
    assert rng_equal(before, rng_state())


def test_statistics_preserve_disconnection_zero_and_sign_denominators():
    p = [torch.ones(3), torch.ones(2)]
    stats = gradient_stats(p, [torch.tensor([-2., 0., 3.]), None])
    assert stats['parameters'] == 5 and stats['connected_values'] == 3
    assert stats['disconnected_parameters'] == 1 and stats['finite']
    assert stats['positive'] == stats['negative'] == stats['zero'] == 1
    assert abs(stats['l2']-np.sqrt(13)) < 1e-12


def test_cli_is_disabled_before_loading_model_or_opening_outputs():
    result = subprocess.run([sys.executable, '-m',
        'variants.staged_reconstruction.joint_revocation_audit.worker'], capture_output=True, text=True)
    assert result.returncode != 0 and 'opt-in' in result.stderr
