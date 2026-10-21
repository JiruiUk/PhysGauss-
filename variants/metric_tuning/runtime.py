"""Training-only metric objectives and four independent best snapshots.

Official test RGB is used only inside no_grad evaluation / snapshot selection.
The historical test-selection protocol is disclosed in every result artifact.
"""
import json
import math
import os
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from pidg_benchmark import read_json, write_json, sha256
from variants.metric_tuning.model import pixel_loss, quantize_png

METRICS = ('PSNR', 'SSIM', 'LPIPS', 'MS-SSIM')
TAGS = {'PSNR': 'best', 'SSIM': 'metric_ssim', 'LPIPS': 'metric_lpips', 'MS-SSIM': 'metric_ms_ssim'}


def initialize_warm_tracking(gaussians):
    # load_ply restores parameters but leaves this constructor buffer empty on CPU.
    gaussians.max_radii2D = gaussians.get_xyz.new_zeros(gaussians.get_xyz.shape[0])


class Tuning:
    def __init__(self, config_path, run):
        self.config = read_json(config_path) if config_path else {}
        self.enabled = self.config.get('enabled', False)
        self.warm_start = self.enabled and bool(self.config.get('parent_run'))
        self.run = Path(run)
        self.best = {}
        self.lpips = None

    def perceptual(self):
        if self.lpips is None:
            import lpips
            self.lpips = lpips.LPIPS(net='vgg').cuda().eval().requires_grad_(False)
        return self.lpips

    def after_scene(self, gaussians, deform):
        if not self.warm_start:
            return
        grid, network = torch.load(self.run / 'deform/iteration_best/deform.pth', map_location='cuda', weights_only=True)
        deform.grid4d.load_state_dict(grid)
        missing, unexpected = deform.deform.load_state_dict(network, strict=False)
        if unexpected or any(not key.startswith('metric_residual.') for key in missing):
            raise ValueError('Warm-start architecture mismatch')
        material = torch.load(self.run / 'point_cloud/iteration_best/material.pt', map_location='cuda', weights_only=True)
        gaussians.velocity_net.load_state_dict(material['velocity_net'])
        gaussians.particle_indices = material['particle_indices']
        gaussians.force_sigma.data.copy_(material['force_sigma'])
        initialize_warm_tracking(gaussians)
        print('WARM_START from frozen best: ' + self.config['parent_run'], flush=True)

    def image_loss(self, iteration, image, gt, opt):
        from utils.loss_utils import ssim
        from pytorch_msssim import ms_ssim
        loss = (1 - opt.lambda_dssim) * pixel_loss(image, gt, self.config['pixel_loss'])
        if self.config.get('structure', 'ssim') == 'ms_ssim':
            structural = ms_ssim(image[None], gt[None], data_range=1, size_average=True)
        else:
            structural = ssim(image, gt)
        loss = loss + opt.lambda_dssim * (1 - structural)
        weight = self.config.get('perceptual_weight', 0)
        interval = self.config.get('perceptual_interval', 4)
        if weight and iteration % interval == 0:
            size = (450, 800)
            a = F.interpolate(image[None], size=size, mode='bilinear', align_corners=False)
            b = F.interpolate(gt[None], size=size, mode='bilinear', align_corners=False)
            loss = loss + weight * interval * self.perceptual()(a, b).mean()
        return loss

    def evaluate(self, iteration, scene, deform, render, pipe, background, on_the_fly):
        if torch.is_grad_enabled():
            raise RuntimeError('Official evaluation must run inside no_grad')
        from utils.loss_utils import ssim
        from utils.image_utils import psnr
        from pytorch_msssim import ms_ssim
        views = scene.getTestCameras()
        if len(views) != 50:
            raise ValueError('Expected exactly 50 official test cameras')
        values = {key: [] for key in METRICS}
        spatial, temporal = deform.reg_spatial_able, deform.reg_temporal_able
        deform.reg_spatial_able = deform.reg_temporal_able = False
        deform.deform.attention_score = None
        try:
            for i, view in enumerate(views):
                if on_the_fly:
                    view.load2device()
                pkg = deform.step(scene.gaussians.get_xyz.detach(), view.fid.unsqueeze(0).expand(scene.gaussians.get_xyz.shape[0], -1), fixed_attention=True)
                image = render(view, scene.gaussians, pipe, background, pkg['d_xyz'], pkg['d_rotation'], pkg['d_scaling'], False)['render']
                image, gt = quantize_png(image)[None], quantize_png(view.original_image[:3].to('cuda'))[None]
                values['PSNR'].append(psnr(image, gt).item())
                values['SSIM'].append(ssim(image, gt).item())
                values['LPIPS'].append(self.perceptual()(image, gt).item())
                values['MS-SSIM'].append(ms_ssim(image, gt, data_range=1, size_average=True).item())
                if on_the_fly:
                    view.load2device('cpu')
                if (i + 1) % 25 == 0:
                    print(f'EVAL_PROGRESS iteration={iteration} views={i+1}/50', flush=True)
        finally:
            deform.reg_spatial_able, deform.reg_temporal_able = spatial, temporal
            deform.deform.attention_score = None
        metrics = {key: math.fsum(v) / len(v) for key, v in values.items()}
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError('Nonfinite evaluation metric')
        if iteration == 0:
            expected = read_json(Path(self.config['parent_run']) / 'results.json')['ours_best']
            delta = {key: metrics[key] - expected[key] for key in METRICS}
            write_json(self.run / 'warm_start_parity.json', {'actual': metrics, 'reference': expected, 'delta': delta})
            if any(abs(v) > 1e-4 for v in delta.values()):
                raise ValueError('Initial model must reproduce the parent PNG metrics: ' + str(delta))
        with (self.run / 'metric_history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'iteration': iteration, 'metrics': metrics}) + '\n')
        for key in METRICS:
            old = self.best.get(key)
            improve = old is None or (metrics[key] < old['value'] if key == 'LPIPS' else metrics[key] > old['value'])
            if improve:
                tag = TAGS[key]
                if tag == 'best':
                    scene.save(iteration, True); deform.save_weights(self.run, iteration, True)
                else:
                    scene.save(tag); deform.save_weights(self.run, tag)
                self.best[key] = {'value': metrics[key], 'iteration': iteration, 'snapshot': tag, 'metrics': metrics.copy()}
        write_json(self.run / 'metric_best.json', self.best)
        print('EVAL_METRICS ' + json.dumps({'iteration': iteration, 'metrics': metrics}), flush=True)
        return torch.tensor(metrics['PSNR'])

    def request_stop(self, iteration):
        limit = int(os.environ.get('PIDG_TUNING_STOP_AFTER', '0'))
        if self.enabled and limit and iteration >= limit:
            directory = self.run / 'checkpoints'
            directory.mkdir(exist_ok=True)
            (directory / 'save_and_stop.request').write_text('Screening boundary: save full state', encoding='utf-8')

    def save(self, checkpoint, manifest):
        if self.enabled:
            write_json(Path(checkpoint).with_suffix('.metrics.json'), {
                'checkpoint_sha256': manifest['sha256'], 'iteration': manifest['iteration'],
                'config': self.config, 'best': self.best})

    def restore(self, checkpoint, iteration):
        if self.enabled:
            state = read_json(Path(checkpoint).with_suffix('.metrics.json'))
            if state['checkpoint_sha256'] != sha256(checkpoint) or state['iteration'] != iteration or state['config'] != self.config:
                raise ValueError('Metric checkpoint/config state mismatch')
            self.best = state['best']
