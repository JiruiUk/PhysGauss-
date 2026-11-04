"""One fixed robust parent expected-depth self-reference, never independent GT."""
import torch


def self_depth_loss(accumulated, alpha, parent_accumulated, parent_alpha):
    assert accumulated.shape == alpha.shape == parent_accumulated.shape == parent_alpha.shape
    target = (parent_accumulated.detach()/parent_alpha.detach().clamp_min(1e-8))
    support = (parent_alpha.detach() > .5) & torch.isfinite(target) & (target > 0)
    if not support.any(): return accumulated.sum()*0
    expected = accumulated/alpha.clamp_min(1e-4)
    return torch.log1p((expected[support]-target[support]).abs()).mean()


def choose_runtime(original_class, original_loss, enabled=False):
    if not enabled: return original_class, original_loss
    from .runtime import protected_class
    def protected_loss(package, target):
        return original_loss(package, target)+.1*package['_self_depth_loss']
    return protected_class(original_class), protected_loss
