"""Opt-in residual capacity; zero initialization preserves a pretrained function."""
import torch
from torch import nn


class ResidualAdapter(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        if width < 1 or hidden < 1:
            raise ValueError('Positive feature dimensions required')
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return h + self.net(h)


def quantize_png(image):
    # torchvision.save_image's uint8 rounding, then official to_tensor scaling.
    return torch.floor(image.clamp(0, 1) * 255 + .5) / 255


def pixel_loss(image, gt, mode):
    delta = image - gt
    if mode == 'l1':
        return delta.abs().mean()
    if mode == 'mse':
        return delta.square().mean()
    if mode == 'charbonnier':
        return (delta.square() + 1e-6).sqrt().mean() - .001
    raise ValueError('Unknown pixel loss: ' + str(mode))
