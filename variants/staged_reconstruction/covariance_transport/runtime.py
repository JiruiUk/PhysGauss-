"""Opt-in scene adapter; original renderer and flags-off scene stay unchanged."""
from types import MethodType, SimpleNamespace
import torch
from ..geometry_recovery.model import bounded_scale
from ..stage2.model import compose
from ..invertible_motion.core import install as install_position
from .core import motion_with_differential, covariance, transport, pack


def renderer_for(original):
    if getattr(original, '_covariance_transport_renderer', False):
        return original
    def render(camera, bank, pipe, *args, **kwargs):
        if hasattr(bank, 'transported_covariance'):
            pipe = SimpleNamespace(**dict(vars(pipe), compute_cov3D_python=True))
        return original(camera, bank, pipe, *args, **kwargs)
    render._covariance_transport_renderer = True
    return render


def with_covariance(bank, packed):
    bank.transported_covariance = packed
    bank.get_covariance = lambda modifier: bank.transported_covariance*modifier**2
    return bank


def dynamic_at(scene, time):
    pool = scene.pools[0]
    moved, j = motion_with_differential(scene.deforms[0].motion, pool.get_xyz,
                                       pool.get_xyz.new_full((len(pool.get_xyz), 1), float(time)))
    scale = bounded_scale(pool._scaling, torch.zeros_like(pool._scaling), pool.spacing, True)
    canonical = covariance(scale, pool.get_rotation)
    current = transport(j, canonical)
    # Scale/rotation attributes are compatibility data only. The marked bank MUST
    # use its precomputed covariance through renderer_for, including dynamic alpha.
    bank = SimpleNamespace(get_xyz=moved+scene.centers[0], get_scaling=scale, get_rotation=pool.get_rotation,
                           get_opacity=pool.get_opacity, get_features=pool.get_features,
                           max_sh_degree=3, active_sh_degree=3)
    return with_covariance(bank, pack(current))


def render_scene(scene, renderer, cam, time):
    from ..train_static import PIPE
    dynamic = scene.dynamic_at(time)
    bank = compose(scene.static, [dynamic])
    static = pack(covariance(scene.static.get_scaling, scene.static.get_rotation))
    with_covariance(bank, torch.cat([static, dynamic.transported_covariance]))
    return renderer_for(renderer)(cam, bank, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)


def install(scene, plan, enabled=False, optimizer=None):
    if not enabled or not plan.get('covariance_transport_enabled', False):
        return None
    if getattr(scene, '_covariance_transport_installed', False):
        raise ValueError('Covariance transport already installed')
    if not plan.get('invertible_motion_enabled', False):
        raise ValueError('This controlled covariance experiment requires its shared invertible position map')
    adapter = install_position(scene, plan, True, optimizer)
    scene.dynamic_at = MethodType(dynamic_at, scene)
    scene.render = MethodType(render_scene, scene)
    scene._covariance_transport_installed = True
    return adapter
