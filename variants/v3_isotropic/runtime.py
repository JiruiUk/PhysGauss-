"""Validation-only V3 with the previously tested rendered-scale isotropy prior."""
import math
import torch
from variants.kinematic_resplit.runtime import Tuning as Parent
from variants.isotropic_tuning.runtime import Tuning as Isotropy
from variants.metric_tuning.runtime import read_json, write_json


class Tuning(Parent):
    def __init__(self, config_path, run):
        super().__init__(config_path, run)
        self.isotropic_weight = float(self.config.get('isotropic_weight', 0.))
        self.isotropic_scope = self.config.get('isotropic_scope', 'rendered')
        self.isotropic_eps = float(self.config.get('isotropic_eps', 1e-8))
        self.isotropic_start = int(self.config.get('isotropic_start', 0))
        if not math.isfinite(self.isotropic_weight) or self.isotropic_weight < 0:
            raise ValueError('Invalid isotropic weight')
        if self.isotropic_scope not in ('rendered', 'canonical'):
            raise ValueError('Invalid isotropic scope')
        if not math.isfinite(self.isotropic_eps) or self.isotropic_eps <= 0 or self.isotropic_start < 0:
            raise ValueError('Invalid isotropic epsilon/start')

    def regularization(self, iteration, gaussians, d_scaling, pipe):
        if iteration <= self.isotropic_start:
            return None
        return Isotropy.regularization(self, iteration, gaussians, d_scaling, pipe)

    def evaluate(self, iteration, scene, deform, render, pipe, background, on_the_fly):
        result = super().evaluate(iteration, scene, deform, render, pipe, background, on_the_fly)
        if iteration == 0 and self.warm_start and not self.config.get('engineering_smoke', False):
            expected = read_json(self.config['parent_run']+'/metric_best.json')['PSNR']['metrics']
            actual = self.best['PSNR']['metrics']
            delta = {k: actual[k]-expected[k] for k in expected}
            write_json(self.run/'parent_validation_parity.json', {'expected': expected, 'actual': actual, 'delta': delta})
            if any(abs(v) > 1e-4 for v in delta.values()):
                raise ValueError('Initial V3 weights do not reproduce parent validation metrics')
        return result
