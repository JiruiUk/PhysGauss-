import json
from pathlib import Path

import pytest
import torch

from variants.association_training.model import anchor_objective, temporal_objective


def test_anchor_preserves_correct_partial_visible_patch_without_full_mask_pressure():
    images = torch.zeros(4, 4, 4, dtype=torch.float64)
    masks = images.bool()
    for k in range(4):
        masks[k, k, :] = True
        images[k, k, :] = .4
    reference = torch.full((4,), .4, dtype=torch.float64)
    result = anchor_objective(images, masks, reference)
    assert result['leakage'] == 0 and result['coverage_floor'] == 0


def test_wrong_object_mass_receives_gradient_and_unknown_pixels_are_not_relabelled():
    images = torch.zeros(4, 3, 3, dtype=torch.float64)
    images[0, 0, 0] = .5
    images.requires_grad_()
    masks = torch.zeros_like(images, dtype=torch.bool)
    masks[0, 2, 2] = True
    loss = anchor_objective(images, masks, torch.ones(4, dtype=torch.float64) * .2)
    (loss['leakage'] + loss['coverage_floor']).backward()
    assert torch.isfinite(images.grad).all() and images.grad[0, 2, 2] < 0


def test_empty_anchor_is_skipped_and_invisible_patch_does_not_get_false_label():
    images = torch.zeros(4, 3, 3, requires_grad=True)
    r = anchor_objective(images, torch.zeros_like(images, dtype=torch.bool), torch.zeros(4))
    assert r['leakage'] == 0 and r['coverage_floor'] == 0


def test_coverage_floor_penalizes_hiding_a_known_visible_bank():
    images = torch.zeros(4, 3, 3, requires_grad=True)
    masks = torch.ones_like(images, dtype=torch.bool)
    r = anchor_objective(images, masks, torch.ones(4) * .4)
    assert r['coverage_floor'] > 0
    r['coverage_floor'].backward()
    assert images.grad.abs().sum() > 0


def test_temporal_objective_zero_for_exact_translation_in_pixel_coordinates():
    a = torch.zeros(4, 5, 6)
    a[0, 2, 2] = .4
    b = torch.zeros_like(a)
    b[0, 2, 3] = .4
    f = torch.zeros(5, 6, 2)
    f[..., 0] = 1
    valid = torch.zeros(5, 6, dtype=torch.bool)
    valid[1:4, 1:4] = True
    assert temporal_objective(a, b, f, valid) < 1e-8


def test_temporal_loss_keeps_group_identity_and_has_finite_both_endpoint_gradients():
    a = torch.zeros(4, 3, 3)
    b = torch.zeros_like(a)
    a[0, 1, 1] = .5
    b[1, 1, 1] = .5
    a.requires_grad_(); b.requires_grad_()
    loss = temporal_objective(a, b, torch.zeros(3, 3, 2), torch.ones(3, 3, dtype=torch.bool))
    assert loss > 0
    loss.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0


def test_invalid_or_occluded_flow_has_no_supervision_or_nan_gradient():
    a = torch.rand(4, 3, 3, requires_grad=True)
    b = torch.rand(4, 3, 3, requires_grad=True)
    loss = temporal_objective(a, b, torch.zeros(3, 3, 2), torch.zeros(3, 3, dtype=torch.bool))
    loss.backward()
    assert loss == 0 and a.grad.abs().sum() == 0 and b.grad.abs().sum() == 0


def test_flags_off_exactly_delegates_without_reading_bank_or_deform(tmp_path, monkeypatch):
    from variants.association_training.runtime import Tuning, Parent
    p = tmp_path / 'config.json'
    p.write_text(json.dumps({'enabled': True, 'association_training_enabled': False}))
    tuning = Tuning(p, tmp_path)
    sentinel = object()
    monkeypatch.setattr(Parent, 'coverage_loss', lambda *a: sentinel)
    assert tuning.coverage_loss(50020, None, None, None, None) is sentinel
    assert not hasattr(tuning, 'association_labels')


def test_original_image_and_physics_methods_are_inherited_unchanged():
    from variants.association_training.runtime import Tuning, Parent
    for method in ['image_loss', 'physics_time', 'time_step', 'regularization', 'kinematic_loss', 'after_update', 'evaluate']:
        assert getattr(Tuning, method) is getattr(Parent, method)
