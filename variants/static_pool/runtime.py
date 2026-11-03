"""Original C1 objectives plus the optional static rendering/sampling policy."""
from pathlib import Path
from variants.color_repair.runtime import Tuning as Parent
from variants.metric_tuning.runtime import read_json, write_json
from .render import pool_for


class Tuning(Parent):
    def after_scene(self, gaussians, deform):
        super().after_scene(gaussians, deform)
        self.pool_gaussians = gaussians

    def restore(self, checkpoint, iteration):
        super().restore(checkpoint, iteration)
        if self.config.get('hard_static_pool', False):
            side = read_json(Path(checkpoint).with_suffix('.metrics.json'))
            if side.get('static_pool_binding_sha256') != self.config['static_pool_binding_sha256']:
                raise ValueError('Checkpoint/static-pool binding mismatch')
            pool_for(self.pool_gaussians, self.config)

    def save(self, checkpoint, manifest):
        super().save(checkpoint, manifest)
        if self.config.get('hard_static_pool', False):
            path = Path(checkpoint).with_suffix('.metrics.json')
            side = read_json(path)
            side['static_pool_binding_sha256'] = self.config['static_pool_binding_sha256']
            write_json(path, side)

    def coverage_loss(self, iteration, view, package, gaussians, warp):
        if self.config.get('record_static_pool_sampling', False):
            self.append('camera_sampling.jsonl', {'iteration': iteration, 'frame': view.image_name,
                                                'time': float(view.fid)})
        return super().coverage_loss(iteration, view, package, gaussians, warp)
