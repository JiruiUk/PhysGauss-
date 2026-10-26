import torch
from variants.coverage_tuning.density import CoverageAccumulator


def test_repeated_same_time_is_one_vote_and_missing_contributors_never_seed():
    state = CoverageAccumulator(4,3)
    ids = torch.tensor([[[0,1],[-1,0]]])
    deficit = torch.ones(1,2,2)
    for _ in range(4): state.observe(0,ids,deficit)
    assert state.votes.sum(1).tolist() == [1,1,0,0]
    assert state.select(min_views=2,limit=4).numel() == 0
    state.observe(1,ids,deficit)
    assert set(state.select(min_views=2,limit=4).tolist()) == {0,1}


def test_exact_state_restores_window_and_rejects_changed_geometry_count():
    a = CoverageAccumulator(4,3)
    a.observe(2,torch.tensor([[[1]]]),torch.ones(1,1,1))
    b = CoverageAccumulator.from_state(a.state_dict())
    assert torch.equal(a.score,b.score) and torch.equal(a.votes,b.votes)
    assert b.select(1,1).tolist() == [1]
    a.reset(6)
    assert len(a.score) == 6 and not a.votes.any()


def test_deficit_threshold_background_and_zero_coverage_are_safe():
    a = CoverageAccumulator(3,3)
    a.observe(0,torch.tensor([[[1,-1,2]]]),torch.tensor([[[0.,.7,.4]]]))
    assert a.select(1,3).tolist() == [2]
