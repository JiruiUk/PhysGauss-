"""Optional component-wise momentum gradient guard; parent flags-off path."""
from variants.unified_flow.runtime import Tuning as Parent
from .balance import balance


class Tuning(Parent):
    def __init__(self, *args):
        super().__init__(*args)
        self.guard = self.enabled and self.config.get('momentum_gradient_guard', False)
        self._reference = None
        if self.guard and (self.stress_source != 'shared' or self.velocity_source != 'trajectory' or not self.material_momentum):
            raise ValueError('Guard experiment requires shared stress and trajectory momentum')

    def image_loss(self, iteration, image, gt, opt):
        value = super().image_loss(iteration, image, gt, opt)
        if self.guard and iteration % 100 == 0 and iteration > self.config.get('momentum_start', 8000):
            self._reference = (iteration, value, float(opt.physics_loss_weight))
        return value

    def momentum_loss(self, iteration, gaussians, deform, fid):
        value = super().momentum_loss(iteration, gaussians, deform, fid)
        if not self.guard or iteration <= self.config.get('momentum_start', 8000):
            return value
        if self._reference is None or self._reference[0] != iteration:
            raise RuntimeError('Guard needs same-iteration image objective')
        _, image, weight = self._reference
        self._reference = None
        groups = {'grid': list(deform.grid4d.parameters()),
            'deformation': [p for n,p in deform.deform.named_parameters() if not n.startswith('shared_stress.')]}
        private = list(deform.deform.shared_stress.parameters())+[gaussians.force_sigma]
        guarded, info = balance(image, value, groups, private,
            float(self.config.get('momentum_shared_ratio', .1)), weight)
        self.append('momentum_gradient_guard.jsonl', {'iteration': iteration,
            'true_momentum_forward': float(value.detach()), 'image_forward': float(image.detach()),
            'physics_weight': weight, **info})
        return guarded
