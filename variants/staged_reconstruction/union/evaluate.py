"""Isolated numeric evaluation; count-free dynamic masks, unchanged image gates."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ..prepare_data import ROOT, read, write, sha
from ..train_static import camera, PIPE
from ..stage2.train import image
from ..model import observation_masks
from .core import dynamic_mask
from .prepare import UNION_REPORT, UNION_RUNS


@torch.no_grad()
def evaluate(renderer, scene, phase, step, split, width=1600, smoke=False, standard=False):
    meta = read(ROOT/f'data/derived/kinematic_resplit/transforms_{split}.json')
    rows = []
    metric = None
    if standard:
        from utils.loss_utils import ssim
        import lpips
        assert (Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth').exists()
        metric = lpips.LPIPS(net='vgg', version='0.1').cuda().eval().requires_grad_(False)
    for frame in meta['frames']:
        rgba = image(frame, width)
        gt = torch.tensor(rgba[..., :3]*rgba[..., 3:4], device='cuda').permute(2, 0, 1)
        cam = camera(frame, meta, width)
        package = scene.render(renderer, cam, frame['time'])
        pred = package['render'].clamp(0, 1)
        dynamic = renderer(cam, scene.dynamic_at(frame['time']), PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
        static, background = observation_masks(rgba, max(2, width//100))
        row = {'frame': int(Path(frame['file_path']).stem),
            'PSNR': float(-10*torch.log10((pred-gt).square().mean().clamp_min(1e-12)))}
        for name, mask in [('dynamic', dynamic_mask(rgba)), ('static', static)]:
            mask = torch.as_tensor(mask, device='cuda')
            row[name+'_PSNR'] = float(-10*torch.log10((pred[:, mask]-gt[:, mask]).square().mean().clamp_min(1e-12))) if mask.any() else None
            alpha = dynamic['alpha'] if name == 'dynamic' else package['alpha']
            row[name+'_alpha'] = float(alpha.squeeze()[mask].mean()) if mask.any() else None
        row['background_alpha'] = float(package['alpha'].squeeze()[torch.as_tensor(background, device='cuda')].mean())
        row['dynamic_leakage_on_static'] = float(dynamic['alpha'].squeeze()[torch.as_tensor(static, device='cuda')].mean())
        if standard:
            rounded = (pred*255).round()/255
            row['PSNR_8bit'] = float(-10*torch.log10((rounded-gt).square().mean().clamp_min(1e-12)))
            row['SSIM_8bit'] = float(ssim(rounded[None], gt[None]))
            row['LPIPS_8bit'] = float(metric(rounded[None]*2-1, gt[None]*2-1).mean())
        rows.append(row)
    metrics = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0] if k != 'frame'}
    result = {'stage': 'S3-U', 'phase': phase, 'iteration': step, 'split': split, 'width': width,
        'count': len(rows), 'metrics': metrics, 'frames': rows, 'standard_VGG_LPIPS': standard,
        'limit': 'Appearance-conditioned union masks; no instance/number or independent geometry truth'}
    directory = UNION_REPORT/('smoke' if smoke else phase)
    write(directory/f'{split}_{step:05d}_{width}{"_standard" if standard else ""}.json', result)
    print('UNION_EVALUATION', phase, step, split, width, metrics, flush=True)
    return result
