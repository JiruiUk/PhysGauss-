"""Real-population qualification must reject empty or unstable anchor ranks."""
import torch


def agreement_plan():
    return {'support_floor': .25, 'top_fraction': .02, 'seed_relative_median_max': .1,
        'seed_relative_p90_max': .25, 'seed_spearman_min': .98, 'seed_top_overlap_min': .8}


def test_agreement_rejects_completely_unobserved_population():
    from variants.staged_reconstruction.uncertainty_motion.observed_audit import independent_agreement
    a = torch.zeros(100, dtype=torch.float64)
    result = independent_agreement(a, a, torch.zeros(100, dtype=torch.bool), agreement_plan())
    assert not result['passed'] and result['active_points'] == 0


def test_agreement_requires_qualified_anchors_even_for_identical_rankings():
    from variants.staged_reconstruction.uncertainty_motion.observed_audit import independent_agreement
    a = torch.arange(1, 101, dtype=torch.float64)
    result = independent_agreement(a, a, torch.zeros(100, dtype=torch.bool), agreement_plan())
    assert not result['passed'] and result['qualified_points'] == 0


def test_disagreeing_top_anchor_ranks_fail_even_when_global_error_is_small():
    from variants.staged_reconstruction.uncertainty_motion.observed_audit import independent_agreement
    a = torch.arange(1, 101, dtype=torch.float64)
    b = a.clone(); b[-4:] = torch.tensor([100., 99., 98., 97.])
    result = independent_agreement(a, b, torch.ones(100, dtype=torch.bool), agreement_plan())
    assert result['relative_difference_median'] == 0 and not result['passed']


def test_agreement_accepts_identical_nonempty_qualified_population():
    from variants.staged_reconstruction.uncertainty_motion.observed_audit import independent_agreement
    a = torch.arange(1, 101, dtype=torch.float64)
    assert independent_agreement(a, a, torch.ones(100, dtype=torch.bool), agreement_plan())['passed']
