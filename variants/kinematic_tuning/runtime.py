"""Optional shared clock and joint velocity / trajectory derivative objective."""
import json
import math
import torch
from variants.metric_tuning.runtime import Tuning as MetricTuning
from variants.kinematic_tuning.model import center_trajectory, trajectory_velocity, consistency_loss
from variants.kinematic_tuning.timeline import configure_scene, physics_time


class Tuning(MetricTuning):
    def __init__(self, config_path, run):
        super().__init__(config_path, run)
        self.weight = float(self.config.get('kinematic_weight', 0.))
        self.shared = self.enabled and self.config.get('shared_physics_time', False)
        self.timeline = self.config.get('timeline', 'official_metadata')
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError('Invalid kinematic weight')
        if self.timeline not in ('official_metadata', 'frame_index'):
            raise ValueError('Unknown timeline')
        if self.enabled and self.weight and not self.shared:
            raise ValueError('Consistency requires shared physics time')

    def configure_scene(self, scene, source):
        timeline = configure_scene(scene, self.config, source)
        if timeline is not None:
            (self.run/'timeline.json').write_text(json.dumps(timeline, indent=2), encoding='utf-8')

    def physics_time(self, fid, train_count):
        return physics_time(fid, train_count, self.shared)

    def time_step(self, train_count):
        return 1/(199 if self.timeline == 'frame_index' else train_count-1) if self.shared else 1/train_count

    def velocities(self, gaussians, deform, fid, iteration=0):
        x = gaussians.get_xyz.detach()
        n = x.shape[0]
        count = min(n, int(self.config.get('kinematic_samples', 256)))
        # Evenly spaced particles; cycle their offset without consuming RNG.
        indices = ((torch.arange(count, device=x.device)*n//count) + iteration//10) % n
        x = x[indices]
        t = fid.reshape(1, 1).expand(count, 1)
        spatial, temporal = deform.reg_spatial_able, deform.reg_temporal_able
        deform.reg_spatial_able = deform.reg_temporal_able = False
        deform.deform.attention_score = None
        try:
            def trajectory(x, t):
                warp = deform.step(x, t, fixed_attention=False)['d_xyz']
                return center_trajectory(x, warp)
            derived = trajectory_velocity(trajectory, x, t, self.config.get('kinematic_dt', .001))
            velocity, _ = gaussians.velocity_net(torch.cat([x, t], -1), gaussians.particle_indices[indices])
        finally:
            deform.reg_spatial_able, deform.reg_temporal_able = spatial, temporal
            deform.deform.attention_score = None
        return velocity, derived

    def kinematic_loss(self, iteration, gaussians, deform, fid):
        if not self.enabled or self.weight == 0:
            return None
        interval = int(self.config.get('kinematic_interval', 10))
        start = int(self.config.get('kinematic_start', 1000))
        if iteration <= start or iteration % interval:
            return None
        ramp = min(1., (iteration-start)/max(1, self.config.get('kinematic_ramp', 1000)))
        predicted, derived = self.velocities(gaussians, deform, fid, iteration)
        loss = consistency_loss(predicted, derived, self.config.get('kinematic_huber_delta', .1))
        if not torch.isfinite(loss): raise ValueError('Nonfinite kinematic loss')
        if iteration % 100 == 0 or self.config.get('audit_gradients', False):
            record = {'iteration': iteration, 'loss': loss.detach().item(),
                      'weighted_loss': self.weight*ramp*interval*loss.detach().item(),
                      'rmse': (predicted-derived).square().mean().sqrt().detach().item(),
                      'predicted_rms': predicted.square().mean().sqrt().detach().item(),
                      'trajectory_rms': derived.square().mean().sqrt().detach().item()}
            self.append('kinematic_loss.jsonl', record)
        if self.config.get('audit_gradients', False):
            if self.config.get('audit_parameters', False) and not getattr(self, '_audited', False):
                groups = {'grid': deform.grid4d, 'deformation': deform.deform, 'velocity': gaussians.velocity_net}
                for name, module in groups.items():
                    params = [p for p in module.parameters() if p.requires_grad]
                    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    finite = all(torch.isfinite(g).all().item() for g in grads if g is not None)
                    magnitude = sum(g.detach().abs().sum().item() for g in grads if g is not None)
                    if not finite or magnitude <= 0: raise ValueError('Missing/nonfinite gradient in '+name)
                    self.append('kinematic_parameter_gradient.jsonl', {'iteration': iteration, 'branch': name,
                                'finite': finite, 'gradient_l1': magnitude})
                self._audited = True
            def hook(name):
                def audit(grad):
                    if not torch.isfinite(grad).all(): raise ValueError('Nonfinite kinematic gradient')
                    self.append('kinematic_gradient.jsonl', {'iteration': iteration, 'branch': name,
                                'gradient_l1': grad.detach().abs().sum().item()})
                return audit
            predicted.register_hook(hook('velocity'))
            derived.register_hook(hook('trajectory'))
        return self.weight*ramp*interval*loss

    def append(self, name, record):
        with (self.run/name).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')

    def diagnostics(self, iteration, scene, deform):
        # Fixed training timestamps and particle subset; no official test RGB.
        values = []
        views = sorted(scene.getTrainCameras(), key=lambda v: v.image_name)
        for index in (0, len(views)//4, len(views)//2, 3*len(views)//4, len(views)-1):
            predicted, derived = self.velocities(scene.gaussians, deform, views[index].fid.to('cuda'))
            error = (predicted-derived).square().mean().sqrt().item()
            speed = derived.square().mean().sqrt().item()
            values.append({'time': views[index].fid.item(), 'rmse': error, 'trajectory_rms': speed,
                           'predicted_rms': predicted.square().mean().sqrt().item(),
                           'relative_rmse': error/max(speed, 1e-8)})
        self.append('kinematic_diagnostics.jsonl', {'iteration': iteration, 'times': values,
                    'rmse': sum(v['rmse'] for v in values)/len(values),
                    'trajectory_rms': sum(v['trajectory_rms'] for v in values)/len(values)})

    def evaluate(self, *args, **kwargs):
        # Kept as a static, reviewable copy of the historical PNG evaluator.
        from variants.kinematic_tuning.evaluation import evaluate
        return evaluate(self, *args, **kwargs)
