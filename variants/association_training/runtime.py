"""Fixed-bank observable identity supervision; original C1 path when disabled."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from variants.color_repair.runtime import Tuning as Parent
from variants.metric_tuning.runtime import read_json, write_json, sha256
from variants.association_training.model import anchor_objective, temporal_objective

ROOT = Path(__file__).resolve().parents[2]


class Tuning(Parent):
    def __init__(self, config_path, run):
        super().__init__(config_path, run)
        self.association_enabled = self.enabled and self.config.get('association_training_enabled', False)

    def after_scene(self, gaussians, deform):
        super().after_scene(gaussians, deform)
        if self.association_enabled:
            self.association_deform = deform

    def configure_scene(self, scene, source):
        super().configure_scene(scene, source)
        if not self.association_enabled:
            return
        path = Path(self.config['association_data'])
        if sha256(path) != self.config['association_data_sha256']:
            raise ValueError('Association labels/data checksum differs')
        with np.load(path) as z:
            self.association_labels = torch.tensor(z['labels'], device='cuda', dtype=torch.long)
            self.anchor_frames = z['anchor_frames'].tolist()
            self.anchor_masks = z['anchor_masks'].copy()
            self.anchor_reference = z['anchor_reference'].copy()
            if str(z['source_checkpoint_sha256']) != self.config['association_source_checkpoint_sha256']:
                raise ValueError('Labels bound to another source checkpoint')
        self.metadata = read_json(Path(source) / 'transforms_train.json')
        self.frames = {int(Path(f['file_path']).stem): f for f in self.metadata['frames']}
        tracks = np.load(ROOT / 'output/surface_support/image_tracks.npz')
        self.components = {int(n): c.copy() for n, c in zip(tracks['frames'], tracks['components'])}
        self.flow_root = ROOT / 'data/derived/flow_comparison_train_only'
        self.flow_rows = read_json(self.flow_root / 'manifest.json')['pairs']
        assert set(self.frames) == {int(v.image_name) for v in scene.getTrainCameras()}
        assert all(n in self.frames for n in self.anchor_frames)

    def group_images(self, gaussians, number):
        from gaussian_renderer import render
        from variants.shape_coherence.observations import camera, PIPE
        from variants.flow_comparison.temporal import warp_at
        if len(gaussians.get_xyz) != len(self.association_labels):
            raise ValueError('Gaussian topology changed after label binding')
        cam = camera(self.frames[number], self.metadata, 1600)
        deform = self.association_deform
        previous = deform.deform.attention_score
        try:
            warp = warp_at(deform, gaussians.get_xyz.detach(), (number - 1) / 199)
        finally:
            deform.deform.attention_score = previous
        images = []
        for offset in [0, 3]:
            color = gaussians.get_xyz.new_zeros((len(self.association_labels), 3))
            for channel in range(min(3, 4 - offset)):
                color[self.association_labels == offset + channel, channel] = 1
            package = render(cam, gaussians, PIPE, gaussians.get_xyz.new_zeros(3),
                             warp['d_xyz'], warp['d_rotation'], warp['d_scaling'], override_color=color)
            count = min(3, 4 - offset)
            images.append(F.avg_pool2d(package['render'][:count], 4))
        return torch.cat(images)

    def association_loss(self, iteration, gaussians):
        from variants.flow_comparison.geometry import forward_backward_valid, sample, pixel_grid
        start = int(self.config.get('association_start', 50000))
        interval = int(self.config.get('association_interval', 20))
        if iteration <= start or (iteration - start) % interval:
            return None
        index = (iteration - start) // interval - 1
        row = self.flow_rows[(index * 37) % len(self.flow_rows)]
        a, b = row['source'], row['target']
        with np.load(self.flow_root / row['file']) as z:
            f = torch.tensor(z['forward'], dtype=torch.float32, device='cuda')
            back = torch.tensor(z['backward'], dtype=torch.float32, device='cuda')
        ma = torch.tensor(self.components[a] >= 0, device='cuda')
        mb = torch.tensor(self.components[b] >= 0, device='cuda')
        vf = forward_backward_valid(f, back) & ma & (sample(mb.float(), pixel_grid(225, 400, f) + f) > .99)
        vb = forward_backward_valid(back, f) & mb & (sample(ma.float(), pixel_grid(225, 400, back) + back) > .99)
        source = self.group_images(gaussians, a)
        target = self.group_images(gaussians, b)
        temporal = .5 * (temporal_objective(source, target, f, vf) + temporal_objective(target, source, back, vb))
        anchor_index = index % len(self.anchor_frames)
        anchor = self.anchor_frames[anchor_index]
        groups = source if anchor == a else target if anchor == b else self.group_images(gaussians, anchor)
        masks = torch.tensor(self.anchor_masks[anchor_index], device='cuda')
        reference = torch.tensor(self.anchor_reference[anchor_index], device='cuda')
        terms = anchor_objective(groups, masks, reference)
        ramp = min(1., (iteration - start) / max(1, int(self.config.get('association_ramp', 200))))
        loss = interval * ramp * (float(self.config.get('association_anchor_weight', .01)) * (terms['leakage'] + terms['coverage_floor'])
                                  + float(self.config.get('association_temporal_weight', .002)) * temporal)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite association objective')
        self.append('association_loss.jsonl', {'iteration': iteration, 'pair': [a, b], 'anchor': anchor,
                    'leakage': float(terms['leakage'].detach()), 'coverage_floor': float(terms['coverage_floor'].detach()),
                    'temporal': float(temporal.detach()), 'weighted_loss': float(loss.detach()),
                    'source_group_mass': float(source.detach().sum()), 'target_group_mass': float(target.detach().sum()),
                    'valid_forward_pixels': int(vf.sum()), 'valid_backward_pixels': int(vb.sum())})
        return loss

    def coverage_loss(self, iteration, view, package, gaussians, warp):
        inherited = super().coverage_loss(iteration, view, package, gaussians, warp)
        if self.config.get('record_association_sampling', False):
            self.append('camera_sampling.jsonl', {'iteration': iteration, 'frame': view.image_name, 'time': float(view.fid)})
        if not self.association_enabled:
            return inherited
        value = self.association_loss(iteration, gaussians)
        if value is None:
            return inherited
        return value if inherited is None else inherited + value

    def save(self, checkpoint, manifest):
        super().save(checkpoint, manifest)
        if self.association_enabled:
            path = Path(checkpoint).with_suffix('.metrics.json')
            state = read_json(path)
            state['association_data_sha256'] = self.config['association_data_sha256']
            write_json(path, state)

    def restore(self, checkpoint, iteration):
        super().restore(checkpoint, iteration)
        if self.association_enabled:
            state = read_json(Path(checkpoint).with_suffix('.metrics.json'))
            if state.get('association_data_sha256') != self.config['association_data_sha256']:
                raise ValueError('Checkpoint association binding differs')
