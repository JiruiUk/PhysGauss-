import torch
from .differential import coordinates, derivative
from .hash_autograd import higher_order_hash


def quaternion_matrix(q):
    assert q.ndim == 2 and q.shape[1] == 4
    w, x, y, z = q.unbind(-1)
    s = 2 / q.square().sum(-1)
    return torch.stack([1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w),
                        s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w),
                        s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)], -1).reshape(-1, 3, 3)


def apply_trajectory_constraints(motion, t):
    from .config import load_config
    if load_config().anchor_deformation:
        identity=motion.new_tensor([1.,0.,0.,0.,0.,0.,0.])
        return identity+t*(motion-identity)
    return motion


def official_position(deform, X, t, higher_order=True):
    """Differentiate the renderer's complete R(q(X,t))X + b(X,t). No TV RNG draws."""
    def evaluate():
        scaled = X * deform.scale_xyz[None]
        time = (t * 2 * deform.grid4d.bound - deform.grid4d.bound) * 0.9
        spatial, temporal = deform.grid4d(torch.cat([scaled, time], -1))
        motion, _, _ = deform.deform(spatial, temporal)
        motion = apply_trajectory_constraints(motion,t)
        if hasattr(deform.deform, "motion_basis"):
            motion = deform.deform.motion_basis.apply_motion(motion, X, t)
        return (quaternion_matrix(motion[:, :4]) @ X[..., None]).squeeze(-1) + motion[:, 4:]
    if higher_order:
        with higher_order_hash():
            return evaluate()
    return evaluate()


def compute_velocity_from_deformation(deformation_model, X, t, create_graph=True):
    X, t = coordinates(X, t)
    x = deformation_model(X, t)
    assert x.shape == X.shape
    v = torch.cat([derivative(x[:, i], t, create_graph) for i in range(3)], dim=-1)
    assert v.shape == X.shape
    return v


def compute_velocity_acceleration(deformation_model, X, t):
    X, t = coordinates(X, t)
    x = deformation_model(X, t)
    assert x.shape == X.shape
    v = torch.cat([derivative(x[:, i], t) for i in range(3)], dim=-1)
    a = torch.cat([derivative(v[:, i], t) for i in range(3)], dim=-1)
    assert v.shape == a.shape == X.shape
    return x, v, a
