"""Matched observed-flow supervision and material-coordinate physics."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
import torch.nn.functional as F
from variants.monotonic_training.runtime import Tuning as Parent
from variants.kinematic_tuning.model import center_trajectory
from . import temporal
from .geometry import flow_loss
from .flow import next_projection, pixel_flow
from .local_momentum import Graph, residual

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/'data/derived/flow_comparison_train_only'


class Tuning(Parent):
    def __init__(self, *args):
        super().__init__(*args)
        self.observed_flow = self.enabled and self.config.get('observed_flow', False)
        self.material_momentum = self.enabled and self.config.get('material_momentum', False)
        self.velocity_source = self.config.get('velocity_source', 'independent')
        self.graph = Graph()
        self._flow_audited = self._momentum_audited = False

    def configure_scene(self, scene, source):
        super().configure_scene(scene, source)
        if not self.observed_flow:
            return
        manifest = json.loads((CACHE/'manifest.json').read_text())
        self.flow_pairs = {p['source']: p for p in manifest['pairs']}
        self.flow_tensors = {}
        actual = {int(v.image_name) for v in scene.getTrainCameras()}
        if actual != set(manifest['train_frame_ids']):
            raise ValueError('Flow provenance and training split differ')
        self.append('flow_contract.jsonl', {'train_frames': len(actual), 'pairs': len(self.flow_pairs),
            'target_resolution': [400, 225], 'depth': 'frozen train-only H3 estimate; not ground truth',
            'velocity_source': self.velocity_source, 'flow_weight': .01,
            'momentum_operator': 'Lagrangian material acceleration, world-space local stress divergence',
            'same_operator_in_both_arms': True})

    def audit(self, name, loss, gaussians, deform, iteration):
        if not self.config.get('audit_parameters', False) or getattr(self, '_'+name+'_audited'):
            return
        modules = {'grid': deform.grid4d, 'deformation': deform.deform,
                   'stress_head': gaussians.velocity_net.stress_head,
                   'velocity_head': gaussians.velocity_net.velocity_head}
        for group, module in modules.items():
            parameters = [p for p in module.parameters() if p.requires_grad]
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True) if parameters else []
            magnitude = sum(float(g.detach().abs().sum()) for g in gradients if g is not None)
            finite = all(bool(g.isfinite().all()) for g in gradients if g is not None)
            required = group in ('grid', 'deformation') or (name == 'momentum' and group == 'stress_head') or \
                (self.velocity_source == 'independent' and group == 'velocity_head')
            if required and (not finite or magnitude <= 0):
                raise ValueError(f'{name}: missing/nonfinite {group} gradient')
            self.append('parameter_gradients.jsonl', {'loss': name, 'group': group,
                'iteration': iteration, 'gradient_l1': magnitude, 'finite': finite, 'required': required})
        setattr(self, '_'+name+'_audited', True)

    def motion_loss(self, iteration, gaussians, deform, view, warp, render, pipe, background):
        pair = self.flow_pairs.get(int(view.image_name))
        start = self.config.get('flow_start', 3000)
        if pair is None or iteration <= start:
            return gaussians.get_xyz.sum()*0
        key = pair['source']
        if key not in self.flow_tensors:
            with np.load(CACHE/pair['file'], allow_pickle=False) as data:
                self.flow_tensors[key] = (torch.from_numpy(data['motion'].copy()), torch.from_numpy(data['valid'].copy()))
        target, valid = (value.to('cuda') for value in self.flow_tensors[key])
        low = SimpleNamespace(image_width=400, image_height=225, FoVx=view.FoVx, FoVy=view.FoVy,
            world_view_transform=view.world_view_transform, full_proj_transform=view.full_proj_transform,
            camera_center=view.camera_center)
        # The raster determines correspondence/visibility only. Analytic Torch
        # projections provide tested gradients in pixel units for future states.
        with torch.no_grad():
            package = render(low, gaussians, pipe, background, warp['d_xyz'].detach(),
                warp['d_rotation'].detach(), warp['d_scaling'].detach(), False,
                override_color=torch.zeros_like(gaussians.get_xyz))
        t = float(view.fid.item()); dt = pair['dt']; future_t = (pair['target']-1)/199
        future = temporal.warp_at(deform, gaussians.get_xyz.detach(), future_t)
        geometric, seen_g = pixel_flow(package, *next_projection(gaussians, future, low))
        v = temporal.velocity(gaussians, deform, gaussians.get_xyz.detach(), gaussians.particle_indices,
                              t, self.velocity_source, self.config.get('velocity_dt', .001))
        current = center_trajectory(gaussians.get_xyz, warp['d_xyz'])
        advected, seen_v = pixel_flow(package, *next_projection(gaussians, future, low, current+dt*v))
        valid = valid & seen_g & seen_v
        geometric_loss, velocity_loss = flow_loss(geometric, target, valid), flow_loss(advected, target, valid)
        raw = .5*(geometric_loss+velocity_loss)
        if not raw.isfinite():
            raise FloatingPointError('Nonfinite observed flow loss')
        ramp = min(1., (iteration-start)/max(1, self.config.get('flow_ramp', 2000)))
        self.append('observed_flow.jsonl', {'iteration': iteration, 'source': pair['source'], 'target': pair['target'],
            'dt': dt, 'geometric_loss': float(geometric_loss.detach()), 'velocity_loss': float(velocity_loss.detach()),
            'valid_pixels': int(valid.sum()), 'ramp': ramp, 'velocity_source': self.velocity_source})
        self.audit('flow', raw, gaussians, deform, iteration)
        return raw*ramp

    def momentum_loss(self, iteration, gaussians, deform, fid):
        start = self.config.get('momentum_start', 8000)
        if iteration <= start:
            return gaussians.get_xyz.sum()*0
        canonical = gaussians.get_xyz.detach()
        all_ids, first, anchors, n_first, n_anchor = self.graph.sample(canonical, iteration,
            count=self.config.get('momentum_samples', 64), k=self.config.get('momentum_neighbors', 12))
        x0 = canonical[all_ids]; ids = gaussians.particle_indices[all_ids]; t = float(fid.item())
        now = temporal.position(deform, x0, t)
        v = temporal.velocity(gaussians, deform, x0, ids, t, self.velocity_source, self.config.get('velocity_dt', .001))
        _, stress = gaussians.velocity_net(torch.cat([x0, x0.new_full((len(x0), 1), t)], -1), ids)
        h = self.config.get('acceleration_dt', 1/199)
        if self.velocity_source == 'trajectory':
            acceleration = temporal.derivative(lambda s: temporal.position(deform, x0, s), t, h, 2)
        else:
            acceleration = temporal.derivative(lambda s: temporal.velocity(gaussians, deform, x0, ids, s, 'independent'), t, h)
        r, valid = residual(now, v, stress, acceleration, gaussians.force_sigma, all_ids, first, anchors, n_first, n_anchor)
        if not valid.any():
            raise ValueError('No usable 3D neighborhoods for momentum residual')
        raw = F.huber_loss(r[valid], torch.zeros_like(r[valid]), delta=1.)
        if not raw.isfinite():
            raise FloatingPointError('Nonfinite material momentum')
        ramp = min(1., (iteration-start)/max(1, self.config.get('momentum_ramp', 4000)))
        self.append('material_momentum.jsonl', {'iteration': iteration, 'raw_huber': float(raw.detach()),
            'residual_rms': float(r[valid].detach().square().mean().sqrt()), 'valid_fraction': float(valid.float().mean()),
            'nodes': len(all_ids), 'anchors': len(anchors), 'ramp': ramp, 'velocity_source': self.velocity_source})
        self.audit('momentum', raw, gaussians, deform, iteration)
        return raw*ramp

    def velocities(self, gaussians, deform, fid, iteration=0):
        predicted, derived = super().velocities(gaussians, deform, fid, iteration)
        return (derived, derived) if self.enabled and self.velocity_source == 'trajectory' else (predicted, derived)

    def diagnostics(self, iteration, scene, deform):
        super().diagnostics(iteration, scene, deform)
        if self.enabled and self.velocity_source == 'trajectory':
            self.append('structural_consistency.jsonl', {'iteration': iteration,
                'zero_velocity_mismatch_by_construction': True, 'not_an_independent_validation_metric': True})

    def after_update(self, iteration, gaussians):
        super().after_update(iteration, gaussians)
        if self.enabled and self.velocity_source == 'trajectory':
            if any(p.requires_grad or p.grad is not None for p in gaussians.velocity_net.velocity_head.parameters()):
                raise ValueError('Unused independent velocity head received gradient')
