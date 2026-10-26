"""Fresh random state, zero optimizer steps and no pretrained-parent loading."""
import json
from types import SimpleNamespace
import pytest
import torch

from variants.v3_iso_scratch.run import configuration, NAMES, validate_initial_state, branch_config
from variants.v3_iso_scratch.runtime import Tuning


def test_scratch_configs_same_except_iso_and_restore_initial_training_schedule():
    a, text = configuration(NAMES[0])
    b, candidate = configuration(NAMES[1])
    branch_config(a, b)
    assert a['parent_run'] == b['parent_run'] == ''
    assert a['isotropic_start'] == b['isotropic_start'] == 0
    assert a['kinematic_weight'] == b['kinematic_weight'] == 0
    assert a['shared_physics_time'] and a['timeline'] == 'frame_index'
    assert text == candidate
    namespace = {}
    exec(text, namespace)
    assert namespace['warm_up'] == 3000
    assert namespace['densify_until_iter'] == 15000
    assert namespace['network_args']['width'] == 256
    assert namespace['network_args']['depth'] == 0


def test_initial_checkpoint_cannot_contain_any_optimizer_updates():
    state = {'iteration': 0, 'gaussian_optimizer': {'state': {}}, 'deform_optimizer': {'state': {}}}
    validate_initial_state(state)
    with pytest.raises(ValueError): validate_initial_state(dict(state, iteration=1))
    with pytest.raises(ValueError):
        validate_initial_state(dict(state, gaussian_optimizer={'state': {0: {'step': torch.tensor(1)}}}))


def test_no_warm_start_and_iso_active_on_first_step(tmp_path):
    config, _ = configuration(NAMES[1])
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    runtime = Tuning(path, tmp_path)
    assert runtime.warm_start is False
    runtime.after_scene(None, None)  # Cannot access any old weight file.
    scales = torch.tensor([[1., 2., 4.]], requires_grad=True)
    runtime.regularization(1, SimpleNamespace(get_scaling=scales), 0.,
                           SimpleNamespace(compute_cov3D_python=False)).backward()
    assert torch.isfinite(scales.grad).all() and scales.grad.abs().sum() > 0


def test_control_no_scale_access_or_rng_work(tmp_path):
    config, _ = configuration(NAMES[0])
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    runtime = Tuning(path, tmp_path)
    before = torch.get_rng_state().clone()
    assert runtime.regularization(1, None, None, None) is None
    assert torch.equal(before, torch.get_rng_state())


def test_branch_refuses_other_model_or_schedule_changes():
    a, _ = configuration(NAMES[0])
    b, _ = configuration(NAMES[1])
    with pytest.raises(ValueError): branch_config(a, dict(b, parent_run='some_old_model'))
    with pytest.raises(ValueError): branch_config(a, dict(b, isotropic_start=1000))


def test_initial_capture_hook_is_flagged_and_before_training_loop():
    from variants.v3_iso_scratch.prepare import prepare, CAPTURE_HOOK
    text = (prepare()/'train_pidg.py').read_text(encoding='utf-8')
    assert text.count(CAPTURE_HOOK) == 1
    assert text.index(CAPTURE_HOOK) < text.index('    for iteration in range(first_iteration, opt.iterations + 1):')
    assert 'tuning.config.get("allow_initial_capture", False)' in CAPTURE_HOOK
