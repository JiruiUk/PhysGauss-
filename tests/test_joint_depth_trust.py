import subprocess
import sys
import torch
from variants.staged_reconstruction.joint_depth_trust.core import self_depth_loss, choose_runtime


def test_normalization_and_teacher_detach_gradcheck():
    d = torch.tensor([1.8, 1.4], dtype=torch.double, requires_grad=True)
    a = torch.tensor([.8, .7], dtype=torch.double, requires_grad=True)
    td = torch.tensor([1.6, 1.68], dtype=torch.double, requires_grad=True)
    ta = torch.tensor([.8, .8], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, y: self_depth_loss(x, y, td, ta), (d, a))
    result = self_depth_loss(d, a, td, ta); result.backward()
    assert td.grad is None and ta.grad is None
    assert d.grad.isfinite().all() and a.grad.isfinite().all()


def test_current_alpha_cannot_drop_support():
    td = torch.tensor([2., 3.]); ta = torch.tensor([1., 1.])
    a = torch.tensor([.001, .8], requires_grad=True); d = torch.tensor([.005, 3.2], requires_grad=True)
    value = self_depth_loss(d, a, td, ta)
    expected = (torch.log1p(torch.tensor(3.))+torch.log1p(torch.tensor(1.)))/2
    assert torch.allclose(value, expected)
    value.backward(); assert d.grad[0] != 0


def test_empty_and_invalid_parent_support_gives_differentiable_zero():
    d = torch.ones(4, requires_grad=True); a = torch.ones(4, requires_grad=True)
    td = torch.tensor([0., -1., float('nan'), 1.]); ta = torch.tensor([1., 1., 1., .5])
    value = self_depth_loss(d, a, td, ta); assert value == 0
    value.backward(); assert torch.equal(d.grad, torch.zeros_like(d))


def test_equal_depth_at_different_opacity_has_zero_loss():
    assert self_depth_loss(torch.tensor([1.]), torch.tensor([.5]), torch.tensor([2.]), torch.tensor([1.])) == 0


def test_off_returns_original_objects_without_runtime_import():
    class Original: pass
    def loss(p, t): return p
    assert choose_runtime(Original, loss) == (Original, loss)


def test_cli_off_no_checkpoint_or_torch_import():
    code = "import runpy,sys; sys.argv=['launch']; runpy.run_module('experiments.staged_reconstruction.S1_joint_depth_trust.launch',run_name='__main__'); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'DISABLED' in result.stdout
