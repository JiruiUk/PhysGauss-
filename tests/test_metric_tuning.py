"""Architecture parity, residual gradients and honest multiobjective selection."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from variants.metric_tuning.model import ResidualAdapter, pixel_loss, quantize_png
from variants.metric_tuning.runtime import Tuning


def network_class(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeformNetwork')
    env = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env['DeformNetwork']


def test_disabled_network_exact_baseline_forward_backward_and_rng(monkeypatch):
    from variants.metric_tuning.prepare import prepare
    repo = prepare()
    baseline = network_class(ROOT / 'third_party/PIDG/scene_PIDG/network.py')
    candidate = network_class(repo / 'scene_PIDG/network.py')
    tensor = torch.tensor
    def cpu_tensor(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return tensor(*args, **kwargs)
    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    torch.manual_seed(7); a = baseline(32, 192, depth=0, width=512)
    state_a = torch.get_rng_state()
    torch.manual_seed(7); b = candidate(32, 192, depth=0, width=512)
    assert torch.equal(state_a, torch.get_rng_state())
    assert a.state_dict().keys() == b.state_dict().keys()
    s, t = torch.randn(9, 32), torch.randn(9, 192)
    left, right = a(s, t), b(s, t)
    assert all(torch.equal(x, y) for x, y in zip(left, right))
    sum(x.square().sum() for x in left).backward()
    sum(x.square().sum() for x in right).backward()
    assert all(torch.equal(x.grad, y.grad) for x, y in zip(a.parameters(), b.parameters()))
    assert sum(p.numel() for p in b.parameters()) == 122894


def test_residual_zero_init_identity_then_learns():
    adapter = ResidualAdapter(32, 12)
    x = torch.randn(10, 32, requires_grad=True)
    assert torch.equal(adapter(x), x)
    opt = torch.optim.Adam(adapter.parameters(), lr=.01)
    for _ in range(2):
        opt.zero_grad(); (adapter(x) - 2 * x.detach()).square().mean().backward(); opt.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    assert adapter.net[0].weight.grad.abs().sum() > 0
    assert not torch.equal(adapter(x), x)


@pytest.mark.parametrize('mode', ['l1', 'mse', 'charbonnier'])
def test_pixel_objectives_have_finite_nonzero_gradients(mode):
    image = torch.full((3, 8, 8), .4, requires_grad=True)
    loss = pixel_loss(image, torch.zeros_like(image), mode)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0


def test_png_quantization_matches_uint8_save_image_rounding():
    x = torch.tensor([-.2, 0., .5/255, 1.5/255, .5, 1., 1.2])
    expected = (x * 255 + .5).clamp(0, 255).to(torch.uint8).float() / 255
    assert torch.equal(quantize_png(x), expected)


def test_metric_checkpoint_rejects_wrong_config_and_unbound_checkpoint(tmp_path):
    from variants.metric_tuning.runtime import sha256
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'enabled': True, 'pixel_loss': 'mse'}))
    tuning = Tuning(path, tmp_path)
    tuning.best = {'PSNR': {'value': 29., 'iteration': 10}}
    checkpoint = tmp_path / 'resume.pt'; checkpoint.write_bytes(b'checkpoint fixture')
    tuning.save(checkpoint, {'sha256': sha256(checkpoint), 'iteration': 10})
    restored = Tuning(path, tmp_path); restored.restore(checkpoint, 10)
    assert restored.best == tuning.best
    restored.config['pixel_loss'] = 'l1'
    with pytest.raises(ValueError, match='mismatch'):
        restored.restore(checkpoint, 10)
    checkpoint.write_bytes(b'different checkpoint')
    with pytest.raises(ValueError, match='mismatch'):
        tuning.restore(checkpoint, 10)


def test_test_metric_evaluation_cannot_run_with_gradients_enabled(tmp_path):
    tuning = Tuning(None, tmp_path)
    assert not tuning.enabled and not tuning.warm_start
    with pytest.raises(RuntimeError, match='no_grad'):
        tuning.evaluate(0, None, None, None, None, None, False)


def runner_module():
    spec = importlib.util.spec_from_file_location('metric_search_runner', ROOT / 'variants/metric_tuning/run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_pareto_keeps_tradeoffs_and_treats_lpips_as_lower_better():
    runner = runner_module()
    def row(name, psnr, lpips):
        return {'id': name, 'metrics': {'PSNR': psnr, 'SSIM': .98, 'LPIPS': lpips, 'MS-SSIM': .97}}
    a, b, dominated = row('psnr', 30, .04), row('lpips', 29, .02), row('bad', 28, .05)
    assert {r['id'] for r in runner.pareto([a, b, dominated])} == {'psnr', 'lpips'}


def test_long_selection_is_bounded_and_preserves_multiobjective_winners(monkeypatch):
    runner = runner_module()
    candidates = []
    for i, key in enumerate(runner.METRICS):
        metric = {'PSNR': 28, 'SSIM': .97, 'LPIPS': .04, 'MS-SSIM': .97}
        metric[key] = .01 if key == 'LPIPS' else metric[key] + .01
        candidates.append({'id': str(i), 'kind': 'candidate', 'additional_iteration': 1000, 'metrics': metric})
    monkeypatch.setattr(runner, 'result_rows', lambda: candidates)
    assert set(runner.select_long()) == {'0', '1', '2', '3'}


def test_warm_start_presets_carry_explicit_flags_and_disable_densification():
    runner = runner_module()
    preset = next(p for p in runner.PRESETS if p['id'] == 'T4_residual_perceptual')
    runtime, text = runner.configuration(preset)
    assert runtime['enabled'] and runtime['perceptual_weight'] > 0
    assert 'metric_residual_width=128' in text and 'densify_until_iter = 0' in text
    runtime, text = runner.configuration(preset, off=True)
    assert not runtime['enabled'] and 'metric_residual_width=0' in text


def test_warm_loaded_radius_buffer_matches_point_count_dtype_and_device():
    from variants.metric_tuning.runtime import initialize_warm_tracking
    class GaussianFixture:
        get_xyz = torch.randn(13, 3, dtype=torch.float64)
        max_radii2D = torch.empty(0)
    g = GaussianFixture()
    initialize_warm_tracking(g)
    assert g.max_radii2D.shape == (13,)
    assert g.max_radii2D.dtype == g.get_xyz.dtype and g.max_radii2D.device == g.get_xyz.device
    assert torch.count_nonzero(g.max_radii2D) == 0


def test_coordinator_import_does_not_confuse_other_variants_prepare_modules(monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, 'prepare', types.ModuleType('unrelated_prepare'))
    runner = runner_module()
    assert runner.SOURCE == ROOT / 'third_party/PIDG'
