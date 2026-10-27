"""An opt-in, train-only alpha centroid objective on an independent PIDG fork."""
import math
import torch
from variants.v3_iso_scratch.runtime import Tuning as Parent
from variants.coverage_tuning.data import frame_paths, read_alpha
from variants.centroid_tuning.model import foreground_centroid, centroid_loss
from variants.metric_tuning.runtime import write_json


class Tuning(Parent):
    def __init__(self, config_path, run):
        super().__init__(config_path, run)
        self.centroid_weight = float(self.config.get('centroid_weight', 0.))
        self.centroid_start = int(self.config.get('centroid_start', 3000))
        self.centroid_ramp = int(self.config.get('centroid_ramp', 2000))
        self.centroid_eps = float(self.config.get('centroid_eps', 1e-6))
        if not math.isfinite(self.centroid_weight) or self.centroid_weight < 0:
            raise ValueError('Centroid weight must be finite and nonnegative')
        if self.centroid_start < 0 or self.centroid_ramp < 1 or not math.isfinite(self.centroid_eps) or self.centroid_eps <= 0:
            raise ValueError('Invalid centroid schedule or denominator floor')
        if any(self.config.get(k, 0) for k in ('mask_weight', 'affine_weight', 'coverage_density', 'isotropic_weight', 'kinematic_weight')):
            raise ValueError('This ablation isolates centroid supervision only')

    def configure_scene(self, scene, source):
        super().configure_scene(scene, source)
        if not self.enabled or self.centroid_weight == 0:
            return
        paths = frame_paths(source, 'train')
        self.train_names = set(paths)
        targets = {}
        for view in scene.getTrainCameras():
            if view.image_name not in paths:
                raise ValueError('Unexpected training view')
            alpha = read_alpha(paths[view.image_name], (view.image_width, view.image_height))
            if alpha.mean() < self.centroid_eps:
                raise ValueError('Empty GT foreground cannot define a centroid')
            view.centroid_target = foreground_centroid(alpha, self.centroid_eps).detach()
            targets[view.image_name] = view.centroid_target.tolist()
        if set(targets) != self.train_names:
            raise ValueError('Incomplete training-centroid membership')
        write_json(self.run / 'centroid_supervision.json', {
            'split': 'train', 'targets_normalized_xy': targets, 'heldout_masks_in_loss': False,
            'meaning': 'whole-scene 2D alpha centroid, not a physical 3D center of mass',
            'object_identity_isolation': False, 'extra_mass_or_mask_loss': False})

    def coverage_loss(self, iteration, view, package, gaussians, warp):
        # Return before touching renderer/GT data when off or before activation.
        if not self.enabled or self.centroid_weight == 0 or iteration <= self.centroid_start:
            return None
        if view.image_name not in self.train_names:
            raise ValueError('Only training targets may enter centroid loss')
        target = view.centroid_target.to(package['alpha'])
        raw = centroid_loss(package['alpha'], target, self.centroid_eps)
        ramp = min(1., (iteration - self.centroid_start) / self.centroid_ramp)
        weighted = self.centroid_weight * ramp * raw
        if not torch.isfinite(weighted):
            raise ValueError('Nonfinite centroid objective')
        if iteration == self.centroid_start + 1 or iteration % 100 == 0:
            predicted = foreground_centroid(package['alpha'].detach(), self.centroid_eps)
            self.append('centroid_loss.jsonl', {
                'iteration': iteration, 'frame': view.image_name, 'loss': raw.detach().item(),
                'weighted_loss': weighted.detach().item(), 'ramp': ramp,
                'predicted_xy': predicted.tolist(), 'target_xy': target.tolist(),
                'distance_normalized': raw.detach().sqrt().item(),
                'mean_alpha': package['alpha'].detach().mean().item()})
            if package['alpha'].requires_grad:
                def audit(gradient):
                    if not torch.isfinite(gradient).all():
                        raise ValueError('Nonfinite centroid alpha gradient')
                    self.append('centroid_gradient.jsonl', {
                        'iteration': iteration, 'alpha_l1': gradient.abs().sum().item()})
                package['alpha'].register_hook(audit)
        return weighted

    def after_update(self, iteration, gaussians):
        # Conventional baseline densification remains entirely in the trainer.
        return None
