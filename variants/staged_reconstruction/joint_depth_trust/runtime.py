"""A process-local wrapper; teacher never joins state_dict or optimizer groups."""
from copy import deepcopy
from types import SimpleNamespace
import torch
from .core import self_depth_loss


def protected_class(original_class):
    class ProtectedScene(original_class):
        def __init__(self, protocol, case, revisable=False):
            super().__init__(protocol, case, revisable)
            teacher = deepcopy(self.static).requires_grad_(False)
            object.__setattr__(self, '_self_depth_teacher', teacher)
            object.__setattr__(self, '_self_depth_cache', {})

        def reference(self, renderer, camera):
            from ..train_static import render_static
            key = (camera.image_width, camera.image_height, camera.FoVx, camera.FoVy,
                camera.world_view_transform.detach().cpu().numpy().tobytes())
            if key not in self._self_depth_cache:
                with torch.no_grad():
                    package = render_static(renderer, camera, self._self_depth_teacher)
                    self._self_depth_cache[key] = (package['depth'].detach(), package['alpha'].detach())
            return self._self_depth_cache[key]

        def render(self, renderer, camera, t):
            from ..train_static import render_static
            package = super().render(renderer, camera, t)
            parent_D, parent_alpha = self.reference(renderer, camera)
            s = self.static
            opacity_only = SimpleNamespace(get_xyz=s.get_xyz.detach(), get_scaling=s.get_scaling.detach(),
                get_rotation=s.get_rotation.detach(), get_opacity=s.get_opacity,
                get_features=s.get_features.detach(), max_sh_degree=3, active_sh_degree=3)
            current = render_static(renderer, camera, opacity_only)
            package['_self_depth_loss'] = self_depth_loss(current['depth'], current['alpha'], parent_D, parent_alpha)
            return package
    return ProtectedScene
