"""Lazy official fixed Small loader. Disabled calls do not import torch/model."""
from pathlib import Path
import sys


def configure():
    import torch
    import cv2
    torch.manual_seed(0)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(root, protocol, enabled=False):
    if not enabled:
        return None
    import torch
    from ..prepare_data import sha
    assets = Path(root) / 'external/depth_anything_v2_small'
    weights = assets / 'depth_anything_v2_vits.pth'
    assert sha(weights) == protocol['weights_sha256']
    sys.path.insert(0, str(assets / 'source'))
    from depth_anything_v2.dpt import DepthAnythingV2
    configure()
    model = DepthAnythingV2(**protocol['model'])
    state = torch.load(weights, map_location='cpu', weights_only=True)
    assert all(torch.isfinite(t).all() for t in state.values())
    model.load_state_dict(state, strict=True)
    model = model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def predict(original, rgb=None, model=None, input_size=518, enabled=False):
    if not enabled:
        return original
    import cv2
    import numpy as np
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    output = model.infer_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), input_size=input_size)
    assert output.shape == rgb.shape[:2] and np.isfinite(output).all() and (output >= 0).all()
    return output.astype(np.float32)
