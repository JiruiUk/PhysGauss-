"""T3 objectives plus an opt-in normalized Gaussian isotropy term."""
import json
import math
import torch
from variants.metric_tuning.runtime import Tuning as MetricTuning
from variants.isotropic_tuning.model import isotropic_normalization


class Tuning(MetricTuning):
    def __init__(self, config_path, run):
        super().__init__(config_path, run)
        self.isotropic_weight = self.config.get('isotropic_weight', 0.)
        self.isotropic_scope = self.config.get('isotropic_scope', 'rendered')
        self.isotropic_eps = self.config.get('isotropic_eps', 1e-8)
        if not math.isfinite(self.isotropic_weight) or self.isotropic_weight < 0:
            raise ValueError('isotropic_weight must be finite and nonnegative')
        if self.isotropic_scope not in ('canonical', 'rendered'):
            raise ValueError('isotropic_scope must be canonical or rendered')
        if not math.isfinite(self.isotropic_eps) or self.isotropic_eps <= 0:
            raise ValueError('isotropic_eps must be finite and positive')

    def regularization(self, iteration, gaussians, d_scaling, pipe):
        # Return before touching Gaussian state or creating any graph/RNG work.
        if not self.enabled or self.isotropic_weight == 0:
            return None
        scales = gaussians.get_scaling
        if self.isotropic_scope == 'rendered':
            # Match the actual PIDG covariance path, including its linear offsets.
            if not pipe.compute_cov3D_python:
                scales = scales + d_scaling
        loss = isotropic_normalization(scales, self.isotropic_eps)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite isotropic normalization loss')
        if iteration == 1 or iteration % 100 == 0:
            values = {'iteration': iteration, 'scope': self.isotropic_scope,
                      'weight': self.isotropic_weight, 'loss': loss.detach().item(),
                      'weighted_loss': self.isotropic_weight * loss.detach().item()}
            with (self.run / 'isotropic_loss.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(values) + '\n')
            if scales.requires_grad:
                def audit(grad):
                    if not torch.isfinite(grad).all():
                        raise ValueError('Nonfinite isotropy scale gradient')
                    record = {'iteration': iteration, 'scale_gradient_l1': grad.detach().abs().sum().item(),
                              'scale_gradient_max': grad.detach().abs().max().item()}
                    with (self.run / 'isotropic_gradient.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(record) + '\n')
                scales.register_hook(audit)
        return self.isotropic_weight * loss
