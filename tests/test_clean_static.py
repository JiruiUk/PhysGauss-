import numpy as np
import pytest
import torch
from variants.staged_reconstruction.clean_static.evidence import ray_evidence, classify, accumulation


K = np.array([[80., 0, 32], [0, 80, 24], [0, 0, 1.]])
PROTOCOL = {'minimum_contradiction_views': 3, 'contradiction_fraction': .6,
            'depth_relative_tolerance': .025, 'depth_absolute_tolerance': .05}


def test_missing_depth_is_unknown_never_empty_background():
    a, b, c, unknown = ray_evidence(np.array([[0., 0, 2]]), np.eye(4), K, np.zeros((48, 64)), np.ones((48, 64), bool))
    assert unknown.all() and not (a|b|c).any()


def test_unreliable_depth_cannot_quarantine():
    a, b, c, unknown = ray_evidence(np.array([[0., 0, 2]]), np.eye(4), K, np.ones((48, 64))*4, np.zeros((48, 64), bool))
    assert unknown.all() and not (a|b|c).any()


def test_behind_surface_is_occluded_not_negative_static_evidence():
    a, b, c, unknown = ray_evidence(np.array([[0., 0, 6]]), np.eye(4), K, np.ones((48, 64))*4, np.ones((48, 64), bool))
    assert c.all() and not (a|b|unknown).any()


def test_visible_surface_and_point_in_measured_free_space_are_distinct():
    a, b, c, unknown = ray_evidence(np.array([[0., 0, 4], [0, 0, 2]]), np.eye(4), K, np.ones((48, 64))*4, np.ones((48, 64), bool))
    assert a.tolist() == [True, False] and b.tolist() == [False, True] and not (c|unknown).any()


def test_minimum_distinct_views_and_contradiction_fraction_both_required():
    assert classify(np.array([0, 0, 20]), np.array([2, 3, 3]), PROTOCOL).tolist() == [False, True, False]


def test_controlled_geometry_preserves_static_surface_and_occluded_surface():
    # Independent analytic geometry GT; this is a mechanism test, not RGB reconstruction.
    xyz = np.array([[0., 0, 4], [.2, 0, 4], [0, 0, 2], [0, 0, 6]])
    views = np.repeat(np.eye(4)[None], 6, 0); views[:, 0, 3] = np.linspace(-.2, .2, 6)
    result = accumulation(xyz, views, K, np.ones((6, 48, 64))*4, np.ones((6, 48, 64), bool), PROTOCOL)
    assert result['quarantine'].tolist() == [False, False, True, False]
    assert result['occluded'][-1] == 6


def test_short_time_cluster_cannot_quarantine_even_with_many_views():
    xyz = np.array([[0.,0,2]])
    views = np.repeat(np.eye(4)[None],6,0)
    protocol = {**PROTOCOL,'minimum_contradiction_time_span':20}
    args = (xyz,views,K,np.ones((6,48,64))*4,np.ones((6,48,64),bool),protocol)
    assert not accumulation(*args,frame_ids=np.arange(6))['quarantine'].any()
    assert accumulation(*args,frame_ids=np.arange(6)*5)['quarantine'].all()


def test_original_row_quarantine_keeps_adam_moments_and_recovers_topology(tmp_path):
    from variants.staged_reconstruction.adaptive_static.model import AdaptiveGaussian, replace_rows
    from variants.staged_reconstruction.train_static import optimizer_for
    import copy
    path = tmp_path/'init.npz'
    np.savez(path,xyz=np.zeros((3,3),np.float32),rgb=np.ones((3,3),np.float32)*.5,
             spacing=np.asarray(.04),point_ids=np.array([0,1,2]))
    model = AdaptiveGaussian(path,device='cpu'); opt = optimizer_for(model)
    sum(p.square().sum() for p in model.parameters()).backward(); opt.step(); opt.zero_grad()
    before = copy.deepcopy(model.state_dict()); adam = copy.deepcopy(opt.state_dict())
    keep = torch.tensor([True,False,True]); replace_rows(model,opt,keep)
    for key,value in before.items(): assert torch.equal(model.state_dict()[key],value[keep])
    for group in opt.param_groups:
        old = adam['state'][next(g['params'][0] for g in adam['param_groups'] if g['name'] == group['name'])]
        current = opt.state[group['params'][0]]
        assert torch.equal(current['step'],old['step'])
        for key in ['exp_avg','exp_avg_sq']: assert torch.equal(current[key],old[key][keep])
    restored = AdaptiveGaussian(path,device='cpu'); restored.load_payload(model.state_dict())
    restored_opt = optimizer_for(restored); restored_opt.load_state_dict(opt.state_dict())
    assert restored.point_ids.tolist() == [0,2]
    sum(p.square().sum() for p in restored.parameters()).backward(); restored_opt.step()
    assert [int(v['step']) for v in restored_opt.state.values()] == [2]*6
    # The full parent remains independently recoverable, including the quarantined row.
    restored.load_payload(before); original_opt = optimizer_for(restored); original_opt.load_state_dict(adam)
    assert restored.point_ids.tolist() == [0,1,2]


@pytest.mark.parametrize('entry',['prepare','train'])
def test_clean_entries_disabled_by_default(entry):
    import subprocess,sys
    result = subprocess.run([sys.executable,'-m','variants.staged_reconstruction.clean_static.'+entry],capture_output=True,text=True)
    assert result.returncode != 0 and 'opt-in' in result.stderr
