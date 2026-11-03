import numpy as np
import pytest
from variants.staged_reconstruction.cross_temporal_static.core import (
    confidence_step, epipolar_errors, join_tracks, weighted_objective)


def test_model_agreement_cannot_invent_geometric_support():
    z = np.zeros((3, 4), np.float32)
    assert not confidence_step(z, z, np.ones_like(z), np.ones_like(z, bool)).any()


def test_independent_geometry_can_expand_missing_render_support():
    c = np.zeros((3, 4), np.float32); g = np.full_like(c, .85)
    for _ in range(3):
        c = confidence_step(c, g, np.zeros_like(c), np.zeros_like(c, bool))
    assert (c >= .5).all() and (c <= g).all()


@pytest.mark.parametrize('production_cap', [.85, .95])
def test_confidence_decreases_and_revokes_inconsistent_candidate(production_cap):
    c = np.full((2, 2), production_cap, np.float32); g = c.copy()
    for _ in range(10):
        c = confidence_step(c, g, np.zeros_like(c), np.ones_like(c, bool))
    assert (c < .5).all()


def test_removed_geometric_evidence_revokes_immediately():
    assert not confidence_step(np.ones((2, 2)), np.zeros((2, 2)), np.ones((2, 2)), np.ones((2, 2), bool)).any()


def test_confidence_update_never_exceeds_geometry():
    g = np.full((2, 2), .7)
    c = confidence_step(np.ones((2, 2)), g, np.ones((2, 2)), np.ones((2, 2), bool))
    assert (c <= g+1e-6).all()


def test_track_merge_cannot_contain_two_features_in_one_camera():
    tracks, conflicts = join_tracks([((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (0, 1))])
    assert conflicts == 1
    assert all(len({i for i, _ in t}) == len(t) for t in tracks)
    assert any(len(t) == 3 for t in tracks)


def test_epipolar_gate_uses_calibrated_camera_motion():
    v0 = np.eye(4); v1 = np.eye(4); v1[0, 3] = -1
    k = np.array([[400, 0, 200], [0, 400, 100], [0, 0, 1]])
    x = np.array([[200., 100.], [210, 110.]])
    y = x-np.array([100, 0])
    assert (epipolar_errors(x, y, v0, v1, k) < 1e-6).all()
    y[:, 1] += 10
    assert (epipolar_errors(x, y, v0, v1, k) > 9).all()


def test_unknown_pixels_receive_no_direct_rgb_or_alpha_gradient():
    import torch
    p = torch.full((3, 3, 3), .5, requires_grad=True)
    a = torch.full((1, 3, 3), .5, requires_grad=True)
    c = torch.zeros((3, 3)); c[0, 0] = .8
    gp, ga = torch.autograd.grad(weighted_objective(p, a, torch.zeros_like(p), c), [p, a])
    assert gp[:, 0, 0].abs().sum() > 0 and ga[0, 0, 0] < 0
    assert gp[:, 1, 1].abs().sum() == ga[0, 1, 1].abs() == 0


def test_empty_confidence_is_zero_loss_without_background_supervision():
    import torch
    p = torch.ones((3, 2, 2), requires_grad=True); a = torch.ones((1, 2, 2), requires_grad=True)
    loss = weighted_objective(p, a, torch.zeros_like(p), torch.zeros((2, 2)))
    assert loss == 0
    loss.backward(); assert not p.grad.any() and not a.grad.any()


@pytest.mark.parametrize('module', ['prepare', 'train'])
def test_disabled_entry_does_not_touch_data_or_renderer(module):
    import subprocess, sys
    p = subprocess.run([sys.executable, '-m', f'variants.staged_reconstruction.cross_temporal_static.{module}'], capture_output=True, text=True)
    assert p.returncode != 0 and 'opt-in' in p.stderr
    assert 'ninja' not in p.stdout
