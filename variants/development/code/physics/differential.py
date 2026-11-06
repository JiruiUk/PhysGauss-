"""Per-particle derivatives; models must not mix particles across batch rows."""
import torch


def derivative(y, x, create_graph=True):
    if not y.requires_grad:
        return x * 0
    result = torch.autograd.grad(y.sum(), x, create_graph=create_graph,
                                 retain_graph=True, allow_unused=True)[0]
    return x * 0 if result is None else result + x * 0


def coordinates(X, t):
    assert X.ndim == 2 and X.shape[1] == 3
    if not X.requires_grad:
        X = X.clone().requires_grad_(True)
    t = torch.as_tensor(t, device=X.device, dtype=X.dtype)
    if t.numel() == 1:
        t = t.reshape(1, 1).expand(len(X), 1).clone()
    if t.shape != (len(X), 1):
        raise ValueError("Time must be scalar or [N,1]")
    if not t.requires_grad:
        t = t.clone().requires_grad_(True)
    return X, t


def jacobian(y, X):
    assert y.shape == X.shape
    result = torch.stack([derivative(y[:, i], X) for i in range(3)], dim=1)
    assert result.shape == (len(X), 3, 3)
    return result
