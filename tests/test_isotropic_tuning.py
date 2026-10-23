"""Isotropy values, scale invariance, gradients and opt-out compatibility."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from variants.isotropic_tuning.model import isotropic_normalization


def test_isotropic_axes_zero_and_anisotropic_axes_positive():
    assert isotropic_normalization(torch.tensor([[2., 2., 2.], [8., 8., 8.]])).item() == 0
    assert isotropic_normalization(torch.tensor([[1., 2., 3.]])).item() == pytest.approx(1/3)


def test_scale_sign_and_axis_order_invariance():
    x = torch.tensor([[.1, .2, .9], [2., 4., 3.]], dtype=torch.float64)
    expected = isotropic_normalization(x)
    for multiplier in (1e-4, 10000., -2.):
        torch.testing.assert_close(isotropic_normalization(x * multiplier), expected)
    torch.testing.assert_close(isotropic_normalization(x[:, [2, 0, 1]]), expected)


def test_finite_at_zero_tiny_and_large_scales():
    x = torch.tensor([[0., 0., 0.], [1e-30, 2e-30, 3e-30], [1e37, 2e37, 3e37]], requires_grad=True)
    value = isotropic_normalization(x)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(x.grad).all()


def test_gradcheck_and_gradient_reduces_axis_inequality():
    x = torch.tensor([[.7, 2., 4.]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(isotropic_normalization, (x,))
    before = isotropic_normalization(x)
    grad, = torch.autograd.grad(before, x)
    assert grad.abs().sum() > 0
    assert isotropic_normalization(x - .1 * grad) < before


def test_penalty_reaches_canonical_scales_and_deformation_head(tmp_path):
    from variants.isotropic_tuning.runtime import Tuning
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'enabled': True, 'isotropic_weight': .001}))
    tuning = Tuning(config, tmp_path)
    log_scale = torch.tensor([[0., .3, .8]], requires_grad=True)
    offset = torch.tensor([[.1, .2, .3]], requires_grad=True)
    g = SimpleNamespace(get_scaling=log_scale.exp())
    loss = tuning.regularization(1, g, offset, SimpleNamespace(compute_cov3D_python=False))
    loss.backward()
    assert torch.isfinite(log_scale.grad).all() and log_scale.grad.abs().sum() > 0
    assert torch.isfinite(offset.grad).all() and offset.grad.abs().sum() > 0
    audit = json.loads((tmp_path / 'isotropic_gradient.jsonl').read_text().splitlines()[0])
    assert audit['scale_gradient_l1'] > 0


@pytest.mark.parametrize('enabled,weight', [(False, 1.), (True, 0.)])
def test_disabled_term_never_accesses_gaussians_and_preserves_rng(tmp_path, enabled, weight):
    from variants.isotropic_tuning.runtime import Tuning
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'enabled': enabled, 'isotropic_weight': weight}))
    tuning = Tuning(path, tmp_path)
    rng = torch.get_rng_state().clone()
    assert tuning.regularization(1, None, None, None) is None
    assert torch.equal(rng, torch.get_rng_state())


def test_canonical_scope_does_not_touch_deformation_offset(tmp_path):
    from variants.isotropic_tuning.runtime import Tuning
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'enabled': True, 'isotropic_weight': .01, 'isotropic_scope': 'canonical'}))
    tuning = Tuning(path, tmp_path)
    x = torch.tensor([[1., 2., 4.]], requires_grad=True)
    tuning.regularization(2, SimpleNamespace(get_scaling=x), None, None).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


@pytest.mark.parametrize('value', [-1., float('inf'), float('nan')])
def test_invalid_weight_rejected(tmp_path, value):
    from variants.isotropic_tuning.runtime import Tuning
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'isotropic_weight': value}))
    with pytest.raises(ValueError):
        Tuning(path, tmp_path)


def test_isotropic_fork_changes_only_optional_training_hook():
    from variants.isotropic_tuning.prepare import prepare
    root = Path(__file__).resolve().parents[1]
    fork = prepare()
    old = (root / 'variants/metric_tuning/code/train_pidg.py').read_text(encoding='utf-8')
    new = (fork / 'train_pidg.py').read_text(encoding='utf-8')
    reverted = new.replace('variants.isotropic_tuning.runtime', 'variants.metric_tuning.runtime')
    hook = ('            iso_loss = tuning.regularization(iteration, gaussians, d_scaling, pipe)\n'
            '            if iso_loss is not None:\n'
            '                loss = loss + iso_loss\n')
    assert reverted.count(hook) == 1
    assert reverted.replace(hook, '') == old


def test_control_and_isotropic_configs_only_differ_by_weight():
    from variants.isotropic_tuning.run import configuration, PRESETS
    control, text = configuration(PRESETS[0])
    for preset in PRESETS[1:]:
        runtime, candidate_text = configuration(preset)
        assert runtime.pop('isotropic_weight') == preset['isotropic_weight']
        assert runtime == {k: v for k, v in control.items() if k != 'isotropic_weight'}
        assert candidate_text == text
    assert control['isotropic_weight'] == 0


def test_reference_table_retains_completed_t3_render_metrics():
    from variants.isotropic_tuning.run import reference_rows
    rows = reference_rows()
    perceptual = next(r for r in rows if r['id'] == 'T3_reference_LPIPS')
    assert perceptual['kind'] == 'reference' and perceptual['snapshot'] == 'metric_lpips'
    assert perceptual['additional_iteration'] == 50000
    assert perceptual['metrics']['LPIPS'] == pytest.approx(.02788456901907921)
