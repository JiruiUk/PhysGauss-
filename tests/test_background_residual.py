"""Reference and gradient behavior, independent of renderer/model fitting."""
import pytest
import torch

from variants.staged_reconstruction.background_residual.core import conditional_static_mask, neutralized_target


def references():
    gt = torch.tensor([[[.1, .7], [0., .4]], [[.1, .6], [0., .4]], [[.8, .1], [0., .4]]])
    background = (gt*.85).clone().requires_grad_(True)
    return gt, background, conditional_static_mask(gt)


def test_disabled_reference_is_same_object_without_touching_static_input():
    gt, _, mask = references()
    assert neutralized_target(gt, None, mask, False) is gt


def test_foreground_empty_and_unknown_references_are_exactly_preserved():
    gt, static, mask = references()
    target = neutralized_target(gt, static, mask, True)
    assert mask.sum() == 1
    assert torch.equal(target[:, ~mask], gt[:, ~mask])
    assert torch.equal(target[:, mask], static[:, mask])
    assert torch.equal(gt[:, 1, 0], torch.zeros(3))


def test_static_reference_is_detached_but_prediction_is_trainable():
    gt, static, mask = references()
    prediction = torch.full_like(gt, .33, requires_grad=True)
    target = neutralized_target(gt, static, mask, True)
    (prediction-target).square().mean().backward()
    assert static.grad is None
    assert prediction.grad is not None and prediction.grad.isfinite().all()
    assert prediction.grad.abs().sum() > 0


def test_background_residual_does_not_pull_an_unchanged_composite():
    gt, static, mask = references()
    prediction = static.detach().clone().requires_grad_(True)
    target = neutralized_target(gt, static, mask, True)
    (prediction[:, mask]-target[:, mask]).abs().mean().backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
    assert (static.detach()[:, mask]-gt[:, mask]).abs().mean() > 0


def test_visible_static_overlay_still_receives_penalty_and_gradient():
    gt, static, mask = references()
    overlay = torch.tensor(.1, requires_grad=True)
    prediction = static.detach()+overlay*mask[None]
    target = neutralized_target(gt, static, mask, True)
    loss = (prediction[:, mask]-target[:, mask]).square().mean()
    loss.backward()
    assert loss.item() > 0 and overlay.grad.item() > 0


def test_dynamic_foreground_gradient_is_unchanged_by_reference_transform():
    gt, static, mask = references()
    pred_a = torch.full_like(gt, .3, requires_grad=True)
    pred_b = pred_a.detach().clone().requires_grad_(True)
    target = neutralized_target(gt, static, mask, True)
    (pred_a[:, ~mask]-gt[:, ~mask]).square().mean().backward()
    (pred_b[:, ~mask]-target[:, ~mask]).square().mean().backward()
    assert torch.equal(pred_a.grad, pred_b.grad)


def test_empty_static_mask_is_identity_in_values():
    gt, static, mask = references()
    target = neutralized_target(gt, static, torch.zeros_like(mask), True)
    assert torch.equal(target, gt)


def test_invalid_shapes_and_nonfinite_reference_are_rejected():
    gt, static, mask = references()
    with pytest.raises(ValueError, match='equal CHW'):
        neutralized_target(gt, static[:, :, :1], mask, True)
    with pytest.raises(ValueError, match='Boolean'):
        neutralized_target(gt, static, mask.float(), True)
    static = static.detach(); static[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        neutralized_target(gt, static, mask, True)
