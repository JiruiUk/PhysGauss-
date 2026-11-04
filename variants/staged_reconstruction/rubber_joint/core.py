"""Default-off real-scene switch; original UnionScene render stays unchanged."""
from copy import deepcopy
from types import SimpleNamespace
import torch
from ..joint_depth_trust.core import self_depth_loss


def configure(scene, enabled=False):
    if not enabled:
        return scene
    assert not any(p.requires_grad for p in scene.static.parameters())
    teacher = deepcopy(scene.static).requires_grad_(False)
    # A teacher/cache is explicitly external to the model and Adam state.
    for name in ('_opacity', '_features_dc', '_features_rest'):
        getattr(scene.static, name).requires_grad_(True)
    return teacher


def depth_term(renderer, scene, teacher, cam):
    from ..train_static import render_static
    with torch.no_grad():
        parent = render_static(renderer, cam, teacher)
    s = scene.static
    opacity_only = SimpleNamespace(get_xyz=s.get_xyz.detach(), get_scaling=s.get_scaling.detach(),
        get_rotation=s.get_rotation.detach(), get_opacity=s.get_opacity,
        get_features=s.get_features.detach(), max_sh_degree=3, active_sh_degree=3)
    current = render_static(renderer, cam, opacity_only)
    return self_depth_loss(current['depth'], current['alpha'], parent['depth'], parent['alpha'])


def objective(renderer, scene, teacher, cam, t, target, enabled=False):
    from ..stage2.train import objective as rgb_loss
    package = scene.render(renderer, cam, t)
    rgb = rgb_loss(package['render'], target)
    self_loss = depth_term(renderer, scene, teacher, cam) if enabled else rgb*0
    return rgb+.1*self_loss, package, rgb, self_loss
