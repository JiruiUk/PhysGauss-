import copy
import subprocess
import sys
from pathlib import Path
import torch
from variants.staged_reconstruction.joint_rgb.core import assert_equivalent
from variants.staged_reconstruction.joint_rgb_stability.core import condition_adam


def test_disabled_identity_exact():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    opt = torch.optim.Adam([p], lr=.002, eps=1e-15)
    before = copy.deepcopy(opt.state_dict()); opts = (opt,)
    assert condition_adam(opts) is opts
    assert_equivalent(before, opt.state_dict(), 0, 0)


def test_epsilon_only_rates_step_and_moments_preserved():
    p = torch.nn.Parameter(torch.tensor([1., 2.])); q = torch.nn.Parameter(torch.tensor([3.]))
    opts = [torch.optim.Adam([p], lr=.002, eps=1e-15), torch.optim.Adam([q], lr=.004, eps=1e-15)]
    for opt, x in zip(opts, [p, q]):
        x.square().sum().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    before = [copy.deepcopy(opt.state_dict()) for opt in opts]
    assert condition_adam(opts, True) is opts
    for previous, opt in zip(before, opts):
        previous['param_groups'][0]['eps'] = 1e-8
        assert_equivalent(previous, opt.state_dict(), 0, 0)


def test_cli_disabled_and_formal_impossible():
    worker = Path(__file__).resolve().parents[1]/'experiments/staged_reconstruction/S1_joint_rgb_stability/run_worker.py'
    for args, text in [([], 'numerical control is disabled'), (['--enable-joint-rgb-stability'], 'Preflight only')]:
        result = subprocess.run([sys.executable, str(worker), *args], capture_output=True, text=True)
        assert result.returncode != 0 and text in result.stderr
