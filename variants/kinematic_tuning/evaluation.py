"""Historical PNG evaluator; separate initialization check for retimed protocol."""
import json
import math
from pathlib import Path
import torch
from variants.metric_tuning.runtime import METRICS, TAGS, read_json, write_json, quantize_png

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
    if iteration == 0 and self.timeline == 'official_metadata':
        expected = read_json(Path(self.config['parent_run']) / 'results.json')['ours_best']
        delta = {key: metrics[key] - expected[key] for key in METRICS}
        write_json(self.run / 'warm_start_parity.json', {'actual': metrics, 'reference': expected, 'delta': delta})
        if any(abs(v) > 1e-4 for v in delta.values()):
            raise ValueError('Initial model must reproduce the parent PNG metrics: ' + str(delta))
    if iteration == 0 and self.timeline == 'frame_index':
        write_json(self.run / 'retimed_initial_metrics.json', {'metrics': metrics, 'note': 'Retimed protocol; historical metric parity is not expected; initial weights copied and hashed'})
    self.diagnostics(iteration, scene, deform)
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
