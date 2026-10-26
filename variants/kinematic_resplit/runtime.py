"""Identical V3/V4 physics, validation-only selection on a frozen global clock."""
import json
import math
import os
from pathlib import Path
import torch
from variants.kinematic_tuning.runtime import Tuning as Parent
from variants.metric_tuning.runtime import METRICS, TAGS, quantize_png, write_json


def validate_scene(scene, source):
    source = Path(source)
    role = json.loads((source/'loader_role.json').read_text())
    ids = []
    for name, bank in [('train', scene.train_cameras), ('test', scene.test_cameras)]:
        metadata = json.loads((source/f'transforms_{name}.json').read_text())
        expected = {Path(f['file_path']).stem: f['time'] for f in metadata['frames']}
        ids.append(set(expected))
        for views in bank.values():
            if {v.image_name for v in views} != set(expected): raise ValueError('Loader membership mismatch')
            for view in views:
                value = (int(view.image_name)-1)/199
                if abs(expected[view.image_name]-value) > 1e-12: raise ValueError('Clock is not globally fixed')
                view.fid = view.fid.new_tensor([value])
    if ids[0] & ids[1]: raise ValueError('Overlapping held-out times')
    return role


def configure_scene_from_env(scene, source):
    path = os.environ.get('PIDG_METRIC_CONFIG')
    if path and json.loads(Path(path).read_text()).get('enabled'): validate_scene(scene, source)


class Tuning(Parent):
    def configure_scene(self, scene, source):
        if not self.enabled: return
        role = validate_scene(scene, source)
        if role['test_slot_semantics'] != 'val': raise ValueError('Training may only load validation into held-out slot')
        self.validation_count = role['heldout_count']
        write_json(self.run/'dataset_role.json', role)

    def evaluate(self, iteration, scene, deform, render, pipe, background, on_the_fly):
        if torch.is_grad_enabled(): raise RuntimeError('Validation must be no_grad')
        from utils.loss_utils import ssim
        from utils.image_utils import psnr
        from pytorch_msssim import ms_ssim
        views = scene.getTestCameras()
        if len(views) != self.validation_count: raise ValueError('Validation count mismatch')
        values = {key: [] for key in METRICS}
        spatial, temporal = deform.reg_spatial_able, deform.reg_temporal_able
        deform.reg_spatial_able = deform.reg_temporal_able = False
        deform.deform.attention_score = None
        try:
            for i, view in enumerate(views):
                if on_the_fly: view.load2device()
                pkg = deform.step(scene.gaussians.get_xyz.detach(), view.fid.unsqueeze(0).expand(scene.gaussians.get_xyz.shape[0], -1), fixed_attention=True)
                image = render(view, scene.gaussians, pipe, background, pkg['d_xyz'], pkg['d_rotation'], pkg['d_scaling'], False)['render']
                image, gt = quantize_png(image)[None], quantize_png(view.original_image[:3].to('cuda'))[None]
                values['PSNR'].append(psnr(image, gt).item())
                values['SSIM'].append(ssim(image, gt).item())
                values['LPIPS'].append(self.perceptual()(image, gt).item())
                values['MS-SSIM'].append(ms_ssim(image, gt, data_range=1, size_average=True).item())
                if on_the_fly: view.load2device('cpu')
                if (i+1) % 10 == 0:
                    print(f'VAL_PROGRESS iteration={iteration} images={i+1}/{len(views)}', flush=True)
        finally:
            deform.reg_spatial_able, deform.reg_temporal_able = spatial, temporal
            deform.deform.attention_score = None
        metrics = {key: math.fsum(v)/len(v) for key, v in values.items()}
        if not all(math.isfinite(v) for v in metrics.values()): raise ValueError('Nonfinite validation metrics')
        self.diagnostics(iteration, scene, deform)
        self.append('metric_history.jsonl', {'iteration': iteration, 'split': 'validation', 'metrics': metrics})
        for key in METRICS:
            old = self.best.get(key)
            if old is None or (metrics[key] < old['value'] if key == 'LPIPS' else metrics[key] > old['value']):
                tag = TAGS[key]
                if tag == 'best':
                    scene.save(iteration, True); deform.save_weights(self.run, iteration, True)
                else:
                    scene.save(tag); deform.save_weights(self.run, tag)
                self.best[key] = {'value': metrics[key], 'iteration': iteration, 'snapshot': tag,
                                  'metrics': metrics.copy(), 'selection_split': 'validation'}
        write_json(self.run/'metric_best.json', self.best)
        print('VAL_METRICS '+json.dumps({'iteration': iteration, 'metrics': metrics}), flush=True)
        return torch.tensor(metrics['PSNR'])
