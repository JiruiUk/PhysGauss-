"""Independent linear image operators with analytic visibility/occlusion truth."""
import itertools
import pytest
import torch
from variants.staged_reconstruction.uncertainty_motion.core import color_vjp, squared_support, recurring_candidates


def render_matrix(a):
    colors = torch.zeros(a.shape[1], 3, dtype=a.dtype, requires_grad=True)
    return (a@colors).T.reshape(3, 2, -1), colors


def test_rademacher_identity_by_complete_sign_enumeration():
    a = torch.tensor([[.4, .2, 0.], [.1, .7, 0.], [0., .2, 0.], [.2, .3, 0.]], dtype=torch.float64)
    rgb, colors = render_matrix(a)
    estimates = [color_vjp(rgb, colors, torch.tensor(s, dtype=a.dtype).reshape(2, 2)).square()
                 for s in itertools.product([-1., 1.], repeat=4)]
    torch.testing.assert_close(torch.stack(estimates).mean(0), a.square().sum(0))


def test_estimate_recovers_occluded_and_invisible_support_without_global_rng_or_grads():
    a = torch.tensor([[.8, .04, 0.], [.5, .025, 0.], [.4, .02, 0.], [.1, .005, 0.]], dtype=torch.float64)
    rgb, colors = render_matrix(a)
    before = torch.get_rng_state().clone()
    result = squared_support(rgb, colors, batches=1024, seed=19)
    torch.testing.assert_close(result['support'], a.square().sum(0), rtol=.08, atol=1e-12)
    assert torch.equal(before, torch.get_rng_state()) and colors.grad is None
    assert result['support'][2] == 0 and result['sampling_se'][2] == 0
    assert result['samples'] == 3072


def test_pixel_weight_uses_square_root_not_squared_weight():
    a = torch.tensor([[.2, .6], [.5, .1], [.3, .2], [.4, .1]], dtype=torch.float64)
    rgb, colors = render_matrix(a)
    weight = torch.tensor([[0., 0.], [.25, .25]], dtype=a.dtype)
    result = squared_support(rgb, colors, batches=1024, seed=17, pixel_weight=weight)
    torch.testing.assert_close(result['support'], (a.square()*weight.flatten()[:, None]).sum(0), rtol=.08, atol=1e-12)
    torch.testing.assert_close(color_vjp(rgb, colors, weight), (a*weight.flatten()[:, None]).sum(0))


def test_private_seed_repeats_exactly_and_independent_seeds_change_estimate():
    rgb, colors = render_matrix(torch.tensor([[.2, .3], [.4, .2], [.7, .2], [.1, .6]], dtype=torch.float64))
    a = squared_support(rgb, colors, batches=32, seed=42)
    b = squared_support(rgb, colors, batches=32, seed=42)
    c = squared_support(rgb, colors, batches=32, seed=43)
    assert torch.equal(a['support'], b['support'])
    assert not torch.equal(a['support'], c['support'])


@pytest.mark.parametrize('weight', [torch.tensor([[-1., 0.], [0., 0.]]), torch.full((2, 2), float('nan')), torch.zeros(3)])
def test_invalid_pixel_weights_rejected(weight):
    rgb, colors = render_matrix(torch.ones(4, 2))
    with pytest.raises(ValueError):
        squared_support(rgb, colors, batches=8, seed=1, pixel_weight=weight)


def test_unobserved_or_unconverged_points_cannot_become_anchors():
    support = torch.tensor([[100., 3., 0., 2.]]).repeat(6, 1)
    valid = torch.ones_like(support, dtype=torch.bool); valid[:, 0] = False
    result = recurring_candidates(support, valid, fraction=.5, minimum_frames=5)
    assert result['candidates'].tolist() == [False, True, False, True]
    assert result['period'].tolist() == [0, 6, 0, 6]


def test_recurrence_does_not_fill_topk_with_zero_or_insufficient_frames():
    support = torch.zeros(6, 4); support[:4, 1] = 1
    result = recurring_candidates(support, torch.ones_like(support, dtype=torch.bool), fraction=1., minimum_frames=5)
    assert result['per_frame'].sum() == 4 and not result['candidates'].any()


def test_selection_is_permutation_equivariant_for_distinct_supports():
    support = torch.tensor([[1., 9., 2., 3.], [4., 8., 1., 2.], [1., 3., 7., 2.]])
    valid = torch.ones_like(support, dtype=torch.bool); permutation = torch.tensor([2, 0, 3, 1])
    a = recurring_candidates(support, valid, fraction=.5, minimum_frames=2)
    b = recurring_candidates(support[:, permutation], valid[:, permutation], fraction=.5, minimum_frames=2)
    assert torch.equal(a['candidates'][permutation], b['candidates'])


def test_ties_use_stable_row_order_and_nonfinite_support_rejected():
    support = torch.ones(5, 4); valid = torch.ones_like(support, dtype=torch.bool)
    assert recurring_candidates(support, valid, fraction=.5)['candidates'].tolist() == [True, True, False, False]
    support[0, 0] = float('inf')
    with pytest.raises(ValueError):
        recurring_candidates(support, valid)
