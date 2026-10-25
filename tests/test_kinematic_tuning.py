"""Shared clocks, full center trajectories and differentiable kinematics."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from variants.kinematic_tuning.model import center_trajectory, trajectory_velocity, consistency_loss
from variants.kinematic_tuning.timeline import build_timeline, apply_timeline, physics_time


def test_full_rotating_translating_trajectory_derivative():
    x = torch.tensor([[1., 0., 2.]], dtype=torch.double)
    t = torch.tensor([[.4]], dtype=torch.double)
    def trajectory(x, t):
        q = torch.cat([(t/2).cos(), t*0, t*0, (t/2).sin()], -1)
        return center_trajectory(x, torch.cat([q, t.expand(-1, 3)**2], -1))
    actual = trajectory_velocity(trajectory, x, t, 1e-4)
    expected = torch.tensor([[-torch.sin(t).item()+.8, torch.cos(t).item()+.8, .8]], dtype=torch.double)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-7)


def test_second_order_endpoints_and_no_out_of_domain_queries():
    t = torch.tensor([[0.], [.0001], [.5], [.9999], [1.]], dtype=torch.double)
    def trajectory(x, time):
        assert ((time >= 0) & (time <= 1)).all()
        return x + time**2
    torch.testing.assert_close(trajectory_velocity(trajectory, torch.zeros(5, 3), t), 2*t.expand(-1, 3))


def test_joint_gradient_reaches_trajectory_and_velocity():
    a = torch.tensor(.3, dtype=torch.double, requires_grad=True)
    b = torch.tensor(.8, dtype=torch.double, requires_grad=True)
    x, t = torch.ones(4, 3, dtype=torch.double), torch.full((4, 1), .4, dtype=torch.double)
    def objective(a, b):
        derived = trajectory_velocity(lambda x, t: x+a*t*t, x, t)
        return consistency_loss(b.expand_as(derived), derived, .1)
    assert torch.autograd.gradcheck(objective, (a, b))
    objective(a, b).backward()
    assert torch.isfinite(a.grad) and a.grad.abs() > 0
    assert torch.isfinite(b.grad) and b.grad.abs() > 0
    assert consistency_loss(x, x, .1) == 0


def test_common_frame_clock_preserves_split_and_uses_filename(tmp_path):
    for split, ids in [('train', [2, 3]), ('test', [1, 4])]:
        (tmp_path / f'transforms_{split}.json').write_text(json.dumps({'frames': [
            {'file_path': f'train/{i:04d}', 'time': j} for j, i in enumerate(ids)]}))
    timeline = build_timeline(tmp_path)
    assert timeline['minimum_frame'] == 1 and timeline['maximum_frame'] == 4
    assert timeline['frames']['0002']['time'] == pytest.approx(1/3)
    view = SimpleNamespace(image_name='0002', fid=torch.tensor([0.]))
    scene = SimpleNamespace(train_cameras={1.: [view]}, test_cameras={1.: []})
    apply_timeline(scene, timeline)
    torch.testing.assert_close(view.fid, torch.tensor([1/3]))


def test_ambiguous_frame_ids_rejected(tmp_path):
    for split in ('train', 'test'):
        (tmp_path / f'transforms_{split}.json').write_text(json.dumps({'frames': [
            {'file_path': 'train/0001', 'time': 0}]}))
    with pytest.raises(ValueError): build_timeline(tmp_path)


def test_legacy_and_shared_physics_time():
    fid = torch.tensor([.75])
    assert torch.equal(physics_time(fid, 150, False), fid.float()/150)
    assert torch.equal(physics_time(fid, 150, True), fid.float())


@pytest.mark.parametrize('enabled,weight', [(False, .001), (True, 0.)])
def test_loss_disabled_no_model_access_or_rng(tmp_path, enabled, weight):
    from variants.kinematic_tuning.runtime import Tuning
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'enabled': enabled, 'kinematic_weight': weight}))
    tuning = Tuning(path, tmp_path)
    before = torch.get_rng_state().clone()
    assert tuning.kinematic_loss(10, None, None, None) is None
    assert torch.equal(before, torch.get_rng_state())


def test_runtime_joint_gradients_and_regularizer_restoration(tmp_path):
    from variants.kinematic_tuning.runtime import Tuning
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'enabled': True, 'shared_physics_time': True, 'kinematic_weight': .001,
                               'kinematic_start': 0, 'kinematic_ramp': 1}))
    tuning = Tuning(path, tmp_path)
    a = torch.nn.Parameter(torch.tensor(.3))
    velocity = torch.nn.Linear(4, 3)
    class Deform:
        reg_spatial_able = reg_temporal_able = True
        deform = SimpleNamespace(attention_score=None)
        def step(self, x, t, fixed_attention=False):
            assert not self.reg_spatial_able and not self.reg_temporal_able
            q = torch.cat([torch.ones_like(t), t*0, t*0, t*0], -1)
            return {'d_xyz': torch.cat([q, a*t.expand(-1, 3)], -1)}
    class Velocity(torch.nn.Module):
        def forward(self, coords, ids): return velocity(coords), coords[:, :3]*0
    g = SimpleNamespace(get_xyz=torch.ones(4, 3), particle_indices=torch.arange(4), velocity_net=Velocity())
    deform = Deform()
    loss = tuning.kinematic_loss(10, g, deform, torch.tensor([.4]))
    loss.backward()
    assert a.grad.abs() > 0 and torch.isfinite(a.grad)
    assert velocity.weight.grad.abs().sum() > 0
    assert deform.reg_temporal_able and deform.reg_spatial_able


def test_unsafe_clock_configuration_rejected(tmp_path):
    from variants.kinematic_tuning.runtime import Tuning
    p = tmp_path/'bad.json'
    p.write_text(json.dumps({'enabled': True, 'kinematic_weight': .001}))
    with pytest.raises(ValueError, match='shared'): Tuning(p, tmp_path)


def test_evaluator_and_fork_hooks_are_importable():
    from variants.kinematic_tuning.evaluation import evaluate
    assert callable(evaluate)
    import ast
    root = Path(__file__).resolve().parents[1]
    for path in (root/'variants/kinematic_tuning').rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
