"""Native depth semantics and finite-difference backward on known geometry."""
from types import SimpleNamespace
import torch
from ..train_static import PIPE


def verify(renderer, cam):
    z = torch.tensor(5., device='cuda', requires_grad=True)
    inv = cam.world_view_transform.T.inverse()

    def render_at(distance):
        camera_point = torch.stack([distance*0, distance*0, distance, distance*0+1])
        xyz = (inv@camera_point)[:3].reshape(1,3)
        point = SimpleNamespace(get_xyz=xyz, get_scaling=xyz.new_full((1,3),.15),
            get_rotation=xyz.new_tensor([[1.,0,0,0]]), get_opacity=xyz.new_full((1,1),.9),
            get_features=xyz.new_zeros((1,16,3)), active_sh_degree=3, max_sh_degree=3)
        p = renderer(cam, point, PIPE, xyz.new_zeros(3), 0.,0.,0., override_color=xyz.new_zeros((1,3)))
        a = p['alpha'].squeeze(); d = p['depth'].squeeze()
        # Fixed, non-edge central pixels; no visibility changes in FD interval.
        h,w = cam.image_height, cam.image_width
        return (d/a.clamp_min(1e-6))[h//2-2:h//2+2,w//2-2:w//2+2].mean()

    value = render_at(z)
    gradient = torch.autograd.grad(value,z)[0]
    with torch.no_grad():
        finite_difference = (render_at(z+.002)-render_at(z-.002))/.004
    assert abs(float(value)-5.) < 2e-4
    assert abs(float(gradient)-1.) < 2e-3
    assert abs(float(gradient-finite_difference)) < 2e-3
    return {'known_camera_z':5., 'normalized_render_z':float(value.detach()),
            'autograd_d_depth_d_z':float(gradient),'finite_difference_d_depth_d_z':float(finite_difference),
            'passed':True}
