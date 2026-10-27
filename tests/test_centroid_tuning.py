"""Specify the image-centroid objective before integration into training."""
import pytest
import torch
import json
from types import SimpleNamespace

from variants.centroid_tuning.model import foreground_centroid, centroid_loss


def test_single_pixel_uses_normalized_xy_coordinates():
    alpha = torch.zeros(1, 5, 9, dtype=torch.float64)
    alpha[0, 1, 6] = 1
    assert torch.allclose(foreground_centroid(alpha), torch.tensor([.75, .25], dtype=torch.float64))


def test_uniform_image_center_and_opacity_scale_invariance():
    alpha = torch.ones(1, 7, 11, dtype=torch.float64)
    assert torch.allclose(foreground_centroid(alpha), torch.tensor([.5, .5], dtype=torch.float64))
    assert torch.allclose(foreground_centroid(alpha), foreground_centroid(alpha * .2))


def test_translation_and_loss_normalization():
    alpha = torch.zeros(1, 5, 9, dtype=torch.float64)
    alpha[0, 2, 6] = 1
    target = torch.tensor([.5, .5], dtype=torch.float64)
    assert centroid_loss(alpha, target).item() == pytest.approx(.25**2)


def test_target_is_detached_and_alpha_gradient_is_finite_nonzero():
    alpha = torch.full((1, 6, 8), .3, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([.7, .4], dtype=torch.float64, requires_grad=True)
    loss = centroid_loss(alpha, target)
    loss.backward()
    assert target.grad is None
    assert torch.isfinite(alpha.grad).all() and alpha.grad.abs().sum() > 0


def test_centroid_gradient_matches_finite_difference():
    alpha = torch.linspace(.1, .9, 30, dtype=torch.float64).reshape(1, 5, 6).requires_grad_()
    target = torch.tensor([.3, .7], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda value: centroid_loss(value, target), (alpha,))


def test_empty_prediction_has_finite_loss_and_gradient():
    alpha = torch.zeros(1, 5, 7, requires_grad=True)
    loss = centroid_loss(alpha, torch.tensor([.5, .5]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(alpha.grad).all()


def test_same_centroid_does_not_imply_same_coverage():
    filled = torch.ones(1, 7, 7)
    ring = filled.clone(); ring[:, 2:5, 2:5] = 0
    assert centroid_loss(ring, foreground_centroid(filled)).item() == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize('value', [torch.zeros(3, 4, 5), torch.ones(1, 1, 3),
                                  torch.full((1, 3, 3), float('nan')),
                                  torch.full((1, 3, 3), -.1)])
def test_invalid_alpha_rejected(value):
    with pytest.raises(ValueError):
        foreground_centroid(value)


def test_invalid_target_rejected():
    with pytest.raises(ValueError):
        centroid_loss(torch.ones(1, 3, 3), torch.tensor([float('nan'), 0.]))


def make_runtime(tmp_path, **extra):
    from variants.centroid_tuning.runtime import Tuning
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(enabled=True, centroid_weight=1., centroid_start=20, centroid_ramp=10, **extra)))
    result = Tuning(config, tmp_path)
    result.train_names = {'0002'}
    return result


def test_runtime_schedule_and_train_only_membership(tmp_path):
    runtime = make_runtime(tmp_path)
    assert runtime.coverage_loss(20, None, None, None, None) is None
    view = SimpleNamespace(image_name='0002', centroid_target=torch.tensor([.7, .4]))
    alpha = torch.ones(1, 4, 5, requires_grad=True) * .4
    loss = runtime.coverage_loss(21, view, {'alpha': alpha}, None, None)
    assert loss.item() == pytest.approx(.1 * (.2**2 + .1**2), rel=1e-5)
    loss.backward()
    with pytest.raises(ValueError, match='Only training'):
        runtime.coverage_loss(25, SimpleNamespace(image_name='0001'), {'alpha': alpha}, None, None)


def test_weight_off_and_global_flags_off_do_not_read_alpha_or_change_rng(tmp_path):
    from variants.centroid_tuning.runtime import Tuning
    for settings in ({'enabled': True, 'centroid_weight': 0}, {'enabled': False, 'centroid_weight': 1}):
        config = tmp_path / 'off.json'; config.write_text(json.dumps(settings))
        runtime = Tuning(config, tmp_path)
        state = torch.get_rng_state().clone()
        assert runtime.coverage_loss(100, None, None, None, None) is None
        runtime.after_update(100, None)
        assert torch.equal(state, torch.get_rng_state())


def test_other_coverage_components_cannot_accidentally_enter_this_ablation(tmp_path):
    with pytest.raises(ValueError, match='isolates centroid'):
        make_runtime(tmp_path, mask_weight=.1)


def test_original_recipe_changes_only_centroid_weight_between_arms():
    from variants.centroid_tuning.run import configuration, NAMES
    control, text = configuration(NAMES[0])
    candidate, other = configuration(NAMES[1])
    assert text == other and 'warm_up = 3000' in text and 'densify_until_iter = 15000' in text
    assert control.pop('centroid_weight') == 0.
    assert candidate.pop('centroid_weight') == 1.
    assert control == candidate
    assert control['perceptual_weight'] == 0 and control['shared_physics_time'] is False
    assert control['parent_run'] == ''
    assert control['kinematic_weight'] == control['isotropic_weight'] == control['mask_weight'] == control['affine_weight'] == 0
    assert control['coverage_density'] is False


def test_fork_only_changes_training_import_and_preserves_parent():
    from variants.centroid_tuning.prepare import prepare
    from variants.coverage_tuning.prepare import prepare as parent
    text = (prepare()/'train_pidg.py').read_text(encoding='utf-8')
    assert text.replace('from variants.centroid_tuning.runtime import Tuning',
                        'from variants.coverage_tuning.runtime import Tuning') == (parent()/'train_pidg.py').read_text(encoding='utf-8')
