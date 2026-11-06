import warnings
import torch
from .differential import coordinates, jacobian


def compute_deformation_gradient(deformation_model, X, t):
    X, t = coordinates(X, t)
    F = jacobian(deformation_model(X, t), X)
    J = torch.linalg.det(F).unsqueeze(-1)
    assert F.shape == (len(X), 3, 3) and J.shape == (len(X), 1)
    if (J.detach() <= 0).any():
        warnings.warn(f"Non-positive deformation Jacobians: {int((J.detach() <= 0).sum())}", RuntimeWarning)
    return F, J
