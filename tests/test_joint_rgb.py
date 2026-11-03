import copy
import random
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch import nn
from variants.staged_reconstruction.joint_rgb.core import (
    open_static_appearance, full_rgb_loss, rng_state, restore_rng, assert_equivalent)


def bank():
    result = nn.Module()
    for name in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest']:
        result.register_parameter(name, nn.Parameter(torch.tensor([.1, .3])))
    return result.requires_grad_(False)


def test_off_is_identity_and_preserves_values_flags_grads_rng():
    value = bank(); state = rng_state(); before = copy.deepcopy(value.state_dict())
    assert open_static_appearance(value) is value
    assert not any(p.requires_grad for p in value.parameters())
    assert_equivalent(before, value.state_dict()); assert_equivalent(state, rng_state())


def test_appearance_only_and_frozen_geometry_adam_state():
    value = bank(); opt = torch.optim.Adam(value.parameters(), lr=.01)
    for p in value.parameters():
        p.requires_grad_(True)
    sum(p.square().sum() for p in value.parameters()).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    value.requires_grad_(False); before = copy.deepcopy(value.state_dict()); moments = copy.deepcopy(opt.state_dict())
    open_static_appearance(value, True)
    assert [n for n, p in value.named_parameters() if p.requires_grad] == ['_opacity', '_features_dc', '_features_rest']
    sum(p.square().sum() for p in value.parameters() if p.requires_grad).backward(); opt.step()
    for i, name in enumerate(['_xyz', '_scaling', '_rotation']):
        assert torch.equal(before[name], value.state_dict()[name])
        assert_equivalent(moments['state'][i], opt.state_dict()['state'][i])
    assert not torch.equal(before['_opacity'], value._opacity)


def test_all_pixels_contribute_without_mask():
    prediction = torch.zeros(3, 4, 5, requires_grad=True); target = torch.ones_like(prediction)
    loss = full_rgb_loss(prediction, target, lambda p, t: 1-(p-t).square().mean())
    assert torch.isclose(loss, torch.tensor(1.))
    loss.backward(); assert (prediction.grad < 0).all()


def test_rng_roundtrip():
    state = rng_state(); x = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng(state); y = (random.random(), np.random.rand(), torch.rand(3))
    assert_equivalent(x, y)


def test_recovery_checks_moments_and_discrete_state():
    a = {'iteration': 20, 'stack': [4, 2], 'Adam': {'exp_avg': torch.tensor([.2])}}
    b = copy.deepcopy(a); assert_equivalent(a, b)
    b['Adam']['exp_avg'] += .01
    with pytest.raises(AssertionError):
        assert_equivalent(a, b)
    b = copy.deepcopy(a); b['stack'].reverse()
    with pytest.raises(AssertionError):
        assert_equivalent(a, b)


def test_cli_disabled_before_native_and_writes():
    result = subprocess.run([sys.executable, '-m', 'variants.staged_reconstruction.joint_rgb.train'], capture_output=True, text=True)
    assert result.returncode != 0 and 'J1 is disabled' in result.stderr
