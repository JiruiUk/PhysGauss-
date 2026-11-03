"""Small independent switch/loss/state helpers; no PIDG changes."""
import random
import numpy as np
import torch

STATIC_APPEARANCE = ('_opacity', '_features_dc', '_features_rest')


def open_static_appearance(static, enabled=False):
    if not enabled:
        return static
    assert not any(p.requires_grad for p in static.parameters())
    for name in STATIC_APPEARANCE:
        getattr(static, name).requires_grad_(True)
    return static


def full_rgb_loss(prediction, target, ssim):
    return .8*(prediction-target).abs().mean()+.2*(1-ssim(prediction, target))


def rng_state():
    return {'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch'])
    if state['rng_cuda']:
        torch.cuda.set_rng_state_all(state['rng_cuda'])


def assert_equivalent(a, b, atol=1e-6, rtol=1e-5, path='state'):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, path
        if a.is_floating_point():
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), path
            assert torch.allclose(a, b, atol=atol, rtol=rtol), path
        else:
            assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_equivalent(a[key], b[key], atol, rtol, path+'/'+str(key))
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_equivalent(x, y, atol, rtol, path+'/'+str(i))
    elif isinstance(a, float):
        assert np.isfinite(a) and np.isclose(a, b, atol=atol, rtol=rtol), path
    else:
        assert a == b, path
