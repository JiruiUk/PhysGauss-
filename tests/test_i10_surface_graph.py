from copy import deepcopy
import numpy as np
import torch
from variants.staged_reconstruction.i10_surface_graph.core import (
    build_patches, build_edges, add_constraints, transition, edge_measurement)
from variants.staged_reconstruction.i10_tracking.io import ROOT, read
from test_i10_revocable_surface import (
    test_first_pass_can_initialize_without_learning_confidence,
    test_unknown_suspends_without_deleting_or_claiming_background,
    test_two_contradictions_revoke_then_two_passes_restore,
    test_unknown_interrupts_consecutive_conflict_count,
    test_state_deepcopy_roundtrip_no_alias,
    test_appearance_missing_is_unknown_not_zero_error,
    test_split_and_frame_checks_reject_confident_misprojection)

P = read(ROOT/'experiments/staged_reconstruction/RubberDuck_I10_surface_graph/protocol.json')


def fixture():
    xyz = np.zeros((12, 5, 3)); xyz[:, :, 2] = 1
    xyz[:, 1, 0] = .02; xyz[:, 2, :2] = [.02, .02]
    xyz[:, 3, 0] = .5; xyz[:, 4, 0] = .03
    xyz[P['fit'], 4, 0] += np.arange(6)*.03
    uv = xyz[:, :, :2]*10+np.array([199.5, 112.])
    evidence = dict(rows=[dict(accepted=True, id='a', start=0, band=0, uv=uv[:, 0].tolist(), binding=dict(row=0))])
    return xyz, uv, evidence


def test_fit_only_nomination_and_held_rejection():
    xyz, uv, evidence = fixture()
    first = build_patches(evidence, xyz, uv, P)
    xyz[P['held'], 1, 0] = .8
    second = build_patches(evidence, xyz, uv, P)
    assert [q['rows'] for q in first] == [q['rows'] for q in second] == [[0], [1], [2]]
    edges = build_edges(second, P)
    bad = next(e for e in edges if {e['i'], e['j']} == {0, 1})
    assert edge_measurement(bad['parent_lengths'], state(), state(), P)['kind'] == 'contradiction'


def test_satellites_are_proxies_never_independent_tracks():
    xyz, uv, evidence = fixture(); nodes = build_patches(evidence, xyz, uv, P)
    assert nodes[0]['is_anchor'] and nodes[0]['members'] == [1, 2]
    assert all(q['target_is_parent_transported_proxy'] and not q['receives_track_auxiliary'] for q in nodes[1:])
    assert len(build_edges(nodes, P)) == 3  # spokes plus local triangle shear support


def test_multiple_owners_remain_bounded_soft_hypotheses():
    xyz, uv, evidence = fixture()
    evidence['rows'] = [dict(evidence['rows'][0], id=str(i)) for i in range(5)]
    nodes = build_patches(evidence, xyz, uv, P)
    for row in [1, 2]:
        values = [q for q in nodes if not q['is_anchor'] and q['rows'][0] == row]
        assert len(values) == 3 and abs(sum(q['membership_weight'] for q in values)-1) < 1e-12


def state():
    return transition(None, dict(kind='pass', passed=True, confidence=.8, frames=[dict(passed=True)]*12), P, True)


def values(frame=0):
    xyz, uv, evidence = fixture(); patches = build_patches(evidence, xyz, uv, P); edges = build_edges(patches, P)
    world = torch.tensor(xyz[0], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[10., 0, 199.5], [0, 10., 112.], [0, 0, 1.]], dtype=world.dtype)
    return world, torch.eye(4, dtype=world.dtype), k, frame, patches, edges, [state() for _ in patches], [state() for _ in edges], P


def test_default_off_exact_identity():
    base = torch.tensor(3., requires_grad=True)
    assert add_constraints(base, *([None]*9))[0] is base


def test_satellite_projection_is_not_an_auxiliary_target():
    args = values(); world = args[0]+torch.tensor([[0., 0, 0], [.003, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    _, track, local, count = add_constraints(args[0].sum()*0, world, *args[1:], True)
    assert count['tracks'] == 1 and track == 0 and local > 0
    local.backward(); assert args[0].grad[1].abs().sum() > 0 and torch.isfinite(args[0].grad).all()


def test_root_suspension_disables_whole_attached_graph():
    args = list(values()); args[6][0]['status'] = 'suspended'
    _, a, b, count = add_constraints(args[0].sum()*0, *args, True)
    assert a == b == 0 and count['tracks'] == count['edges'] == 0


def test_held_outside_frames_no_auxiliary_or_graph_extrapolation():
    for frame in [1, 12, 119]:
        args = values(frame); _, a, b, count = add_constraints(args[0].sum()*0, *args, True)
        assert a == b == 0 and count['tracks'] == count['edges'] == 0


def test_local_lengths_rotation_invariant_and_triangle_shear_sensitive():
    args = values(); world = args[0]
    _, _, original, _ = add_constraints(world.sum()*0, *args, True)
    rotation = world.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    _, _, rotated, _ = add_constraints(world.sum()*0, world@rotation, *args[1:], True)
    assert torch.allclose(original, rotated, atol=1e-12)
    deformation = torch.eye(3, dtype=world.dtype); deformation[0, 1] = .1
    _, _, shear, _ = add_constraints(world.sum()*0, world@deformation, *args[1:], True)
    assert shear > original
