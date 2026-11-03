"""Temporary autograd permissions with caller values, gradients and RNG restored."""
from contextlib import contextmanager
import hashlib
import random
import numpy as np
import torch

STATIC_FIELDS = ('_opacity', '_features_dc', '_features_rest')


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def rng_equal(a, b):
    return (a[0] == b[0] and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
        and a[1][2:] == b[1][2:] and torch.equal(a[2], b[2])
        and ((a[3] is None and b[3] is None) or
             (a[3] is not None and b[3] is not None and len(a[3]) == len(b[3])
              and all(torch.equal(x, y) for x, y in zip(a[3], b[3])))))


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


@contextmanager
def audit_scope(scene, enabled=False):
    if not enabled:
        yield scene
        return
    params = list(scene.parameters())
    flags = [p.requires_grad for p in params]
    gradients = [p.grad for p in params]
    gradient_values = [None if g is None else g.detach().clone() for g in gradients]
    values = {name: value.detach().cpu().clone() for name, value in scene.state_dict().items()}
    before_rng = rng_state()
    try:
        scene.static.requires_grad_(False)
        for name in STATIC_FIELDS:
            getattr(scene.static, name).requires_grad_(True)
        yield scene
    finally:
        with torch.no_grad():
            for name, value in scene.state_dict(keep_vars=True).items():
                saved = values[name].to(value.device)
                if not torch.equal(value, saved):
                    value.copy_(saved)
            for param, flag, grad, saved in zip(params, flags, gradients, gradient_values):
                param.requires_grad_(flag)
                if grad is not None:
                    grad.copy_(saved)
                param.grad = grad
        restore_rng(before_rng)


def gradient_stats(parameters, gradients):
    values = [g.detach().cpu().double().numpy().reshape(-1) for g in gradients if g is not None]
    total = sum(p.numel() for p in parameters)
    if not values:
        return {'parameters': total, 'connected_values': 0, 'disconnected_parameters': len(parameters),
            'finite': True, 'l2': 0., 'mean_abs': 0., 'positive': 0, 'negative': 0, 'zero': 0}
    flat = np.concatenate(values)
    return {'parameters': total, 'connected_values': int(len(flat)),
        'disconnected_parameters': sum(g is None for g in gradients),
        'finite': bool(np.isfinite(flat).all()), 'l2': float(np.linalg.norm(flat)),
        'mean_abs': float(np.abs(flat).mean()), 'positive': int((flat > 0).sum()),
        'negative': int((flat < 0).sum()), 'zero': int((flat == 0).sum())}
