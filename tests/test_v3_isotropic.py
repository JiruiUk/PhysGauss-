"""V3+iso integration: shared clock, validation selection and optional shape prior."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from variants.v3_isotropic.runtime import Tuning


def tuning(tmp_path, **options):
    config = {'enabled': True, 'shared_physics_time': True, 'timeline': 'frame_index',
              'kinematic_weight': 0., 'isotropic_weight': .001, 'isotropic_start': 20}
    config.update(options)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    return Tuning(path, tmp_path)


@pytest.mark.parametrize('options,iteration', [({'enabled': False}, 100), ({'isotropic_weight': 0.}, 100), ({}, 20)])
def test_disabled_or_common_prefix_no_access_and_no_rng(tmp_path, options, iteration):
    runtime = tuning(tmp_path, **options)
    before = torch.get_rng_state().clone()
    assert runtime.regularization(iteration, None, None, None) is None
    assert torch.equal(before, torch.get_rng_state())


def test_iso_reaches_both_scale_branches_and_keeps_kinematic_off(tmp_path):
    runtime = tuning(tmp_path)
    log_scale = torch.tensor([[0., .3, .8]], requires_grad=True)
    offset = torch.tensor([[.1, .2, .3]], requires_grad=True)
    g = SimpleNamespace(get_scaling=log_scale.exp())
    runtime.regularization(100, g, offset, SimpleNamespace(compute_cov3D_python=False)).backward()
    assert log_scale.grad.abs().sum() > 0 and torch.isfinite(log_scale.grad).all()
    assert offset.grad.abs().sum() > 0 and torch.isfinite(offset.grad).all()
    assert runtime.kinematic_loss(100, None, None, None) is None
    assert runtime.physics_time(torch.tensor([.7]), 120).item() == pytest.approx(.7)


def test_configs_only_differ_by_iso_weight(tmp_path):
    from variants.v3_isotropic.run import configuration, NAMES, branch_config
    a, text = configuration(NAMES[0])
    b, candidate = configuration(NAMES[1])
    branch_config(a, b)
    assert text == candidate
    assert a['kinematic_weight'] == b['kinematic_weight'] == 0
    assert a['parent_run'] == b['parent_run']
    with pytest.raises(ValueError): branch_config(a, dict(b, kinematic_weight=.001))


def test_new_fork_changes_only_optional_training_hook():
    from variants.v3_isotropic.prepare import prepare
    root = Path(__file__).resolve().parents[1]
    source = (root/'variants/kinematic_resplit/code/train_pidg.py').read_text(encoding='utf-8')
    fork = (prepare()/'train_pidg.py').read_text(encoding='utf-8')
    reverted = fork.replace('variants.v3_isotropic.runtime', 'variants.kinematic_resplit.runtime')
    hook = ('            iso_loss = tuning.regularization(iteration, gaussians, d_scaling, pipe)\n'
            '            if iso_loss is not None:\n'
            '                loss = loss + iso_loss\n')
    assert reverted.count(hook) == 1
    assert reverted.replace(hook, '') == source
