"""Same cubic hash interpolation, using ordinary Torch ops for higher derivatives.

The released custom backward returns no derivative with respect to input coordinates
in its second backward. It cannot compute complete trajectory Hessians. This path
shares the original embeddings; the RGB renderer keeps the original CUDA path.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import torch

_higher_order = ContextVar("pidg_higher_order", default=False)


@contextmanager
def higher_order_hash():
    token = _higher_order.set(True)
    try:
        yield
    finally:
        _higher_order.reset(token)


def enabled():
    return _higher_order.get()


def interpolate_hash(inputs, embeddings, offsets, per_level_scale, base_resolution):
    """Inputs [B,D] in [0,1]; matches unsigned 32-bit CUDA indexing."""
    B, D = inputs.shape
    L = len(offsets) - 1
    level = torch.arange(L, device=inputs.device, dtype=per_level_scale.dtype)
    scale = torch.exp2(level[:, None] * torch.log2(per_level_scale)[None]) * base_resolution - 1
    resolution = torch.ceil(scale).to(torch.int64) + 1
    pos = inputs[None] * scale[:, None].to(inputs.dtype)
    lower = torch.floor(pos).to(torch.int64)
    frac = pos - lower.to(inputs.dtype)
    smooth = frac.square() * (3 - 2*frac)
    corners = ((torch.arange(2**D, device=inputs.device)[:, None] >>
                torch.arange(D, device=inputs.device)) & 1)
    grid = lower[:, :, None] + corners[None, None]
    weights = torch.where(corners[None, None].bool(), smooth[:, :, None], 1-smooth[:, :, None]).prod(-1)
    stride = torch.cumprod(torch.cat([torch.ones(L, 1, device=inputs.device, dtype=torch.int64),
                                     resolution[:, :-1]], dim=-1), dim=-1)
    dense = (grid * stride[:, None, None]).sum(-1)
    primes = inputs.new_tensor([1, 2654435761, 805459861, 3674653429,
                                2097192037, 1434869437, 2165219737][:D], dtype=torch.int64)
    hashed = torch.zeros_like(dense)
    for d in range(D):
        hashed = torch.bitwise_xor(hashed, (grid[..., d] * primes[d]) & 0xffffffff)
    counts = (offsets[1:] - offsets[:-1]).to(torch.int64)
    index = torch.where((resolution.prod(-1) > counts)[:, None, None], hashed, dense)
    index = index.remainder(counts[:, None, None]) + offsets[:-1, None, None]
    values = embeddings[index.to(torch.int64)]
    result = (values * weights[..., None]).sum(2).permute(1, 0, 2).reshape(B, -1)
    inside = ((inputs >= 0) & (inputs <= 1)).all(-1, keepdim=True)
    return torch.where(inside, result, torch.zeros_like(result))
