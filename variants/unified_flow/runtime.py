"""Matched corrected optical flow; separate vs deformation-shared stress."""
import json
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from variants.flow_comparison.runtime import Tuning as Parent
from variants.monotonic_training.runtime import Tuning as ClockParent
from variants.flow_comparison import temporal
from variants.flow_comparison.geometry import flow_loss
from variants.flow_comparison.local_momentum import residual
from variants.kinematic_tuning.model import center_trajectory
from .prepare_data import CACHE
from .flow import pixel_flow, next_projection
from .model import predict_stress


class Tuning(Parent):
    def __init__(self, *args):
        super().__init__(*args)
        self.raw_flow = self.enabled and self.config.get('raw_flow', False)
        self.stress_source = self.config.get('stress_source', 'independent')

    def configure_scene(self, scene, source):
        if not self.raw_flow:
            return super().configure_scene(scene, source)
        ClockParent.configure_scene(self, scene, source)
        manifest = json.loads((CACHE/'manifest.json').read_text())
        self.flow_pairs = {p['source']: p for p in manifest['pairs']}; self.flow_tensors = {}
        self.train_cameras = {int(c.image_name): c for c in scene.getTrainCameras()}
        assert set(self.train_cameras) == set(manifest['train_frame_ids'])
        self.append('flow_contract.jsonl', {'target': 'raw source->target camera flow, no estimated depth',
            'camera': 'actual scene cameras, preserved loader FoV convention',
            'metadata': 'negative x_mu, recomputed pre-update compositing weights', 'stress_source': self.stress_source})

    def audit(self, name, loss, gaussians, deform, iteration):
        if not self.config.get('audit_parameters', False) or getattr(self, '_'+name+'_audited'):
            return
        modules = {'grid': deform.grid4d, 'deformation': deform.deform,
            'stress_head': deform.deform.shared_stress if self.stress_source == 'shared' else gaussians.velocity_net.stress_head,
            'legacy_velocity': gaussians.velocity_net.velocity_head}
        for group, module in modules.items():
            params = [p for p in module.parameters() if p.requires_grad]
            gradients = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True) if params else []
            magnitude = sum(float(g.detach().abs().sum()) for g in gradients if g is not None)
            finite = all(bool(g.isfinite().all()) for g in gradients if g is not None)
            required = group in ('grid', 'deformation') or (name == 'momentum' and group == 'stress_head')
            if required and (not finite or magnitude <= 0):
                raise ValueError(f'{name}: invalid {group} gradient')
            self.append('parameter_gradients.jsonl', {'loss': name, 'group': group, 'iteration': iteration,
                'gradient_l1': magnitude, 'finite': finite, 'required': required, 'stress_source': self.stress_source})
        setattr(self, '_'+name+'_audited', True)

    def motion_loss(self, iteration, gaussians, deform, view, warp, render, pipe, background):
        if not self.raw_flow:
            return super().motion_loss(iteration, gaussians, deform, view, warp, render, pipe, background)
        pair = self.flow_pairs.get(int(view.image_name)); start = self.config.get('flow_start', 3000)
        if pair is None or iteration <= start:
            return gaussians.get_xyz.sum()*0
        key = pair['source']
        if key not in self.flow_tensors:
            with np.load(CACHE/pair['file']) as data:
                self.flow_tensors[key] = tuple(torch.from_numpy(data[k].copy()) for k in ['observed', 'valid'])
        target, valid = [x.to('cuda') for x in self.flow_tensors[key]]
        def low(camera):
            return SimpleNamespace(image_width=400, image_height=225, FoVx=camera.FoVx, FoVy=camera.FoVy,
                world_view_transform=camera.world_view_transform.cuda(), full_proj_transform=camera.full_proj_transform.cuda(),
                camera_center=camera.camera_center.cuda())
        current_camera, future_camera = low(view), low(self.train_cameras[pair['target']])
        with torch.no_grad():
            package = render(current_camera, gaussians, pipe, background, warp['d_xyz'].detach(),
                warp['d_rotation'].detach(), warp['d_scaling'].detach(), False, override_color=torch.zeros_like(gaussians.get_xyz))
        t = float(view.fid.item()); future_t = (pair['target']-1)/199
        future = temporal.warp_at(deform, gaussians.get_xyz.detach(), future_t)
        geometric, seen = pixel_flow(package, *next_projection(gaussians, future, future_camera), gaussians.get_opacity)
        v = temporal.velocity(gaussians, deform, gaussians.get_xyz.detach(), gaussians.particle_indices, t, 'trajectory', self.config.get('velocity_dt', .001))
        current = center_trajectory(gaussians.get_xyz, warp['d_xyz'])
        advected, _ = pixel_flow(package, *next_projection(gaussians, future, future_camera, current+pair['dt']*v), gaussians.get_opacity)
        valid = valid & seen
        if not valid.any(): raise ValueError('No flow supervision coverage')
        geometry_loss, velocity_loss = flow_loss(geometric, target, valid), flow_loss(advected, target, valid)
        raw = .5*(geometry_loss+velocity_loss)
        if not raw.isfinite(): raise FloatingPointError('Nonfinite raw flow')
        ramp = min(1., (iteration-start)/max(1, self.config.get('flow_ramp', 2000)))
        self.append('observed_flow.jsonl', {'iteration': iteration, 'source': pair['source'], 'target': pair['target'],
            'dt': pair['dt'], 'geometric_loss': float(geometry_loss.detach()), 'velocity_loss': float(velocity_loss.detach()),
            'valid_pixels': int(valid.sum()), 'ramp': ramp, 'velocity_source': 'trajectory', 'stress_source': self.stress_source})
        self.audit('flow', raw, gaussians, deform, iteration)
        return raw*ramp

    def momentum_loss(self, iteration, gaussians, deform, fid):
        if self.stress_source != 'shared':
            return super().momentum_loss(iteration, gaussians, deform, fid)
        start = self.config.get('momentum_start', 8000)
        if iteration <= start: return gaussians.get_xyz.sum()*0
        canonical = gaussians.get_xyz.detach()
        all_ids, first, anchors, n_first, n_anchor = self.graph.sample(canonical, iteration,
            count=self.config.get('momentum_samples', 64), k=self.config.get('momentum_neighbors', 12))
        x = canonical[all_ids]; ids = gaussians.particle_indices[all_ids]; t = float(fid.item())
        now = temporal.position(deform, x, t)
        v = temporal.velocity(gaussians, deform, x, ids, t, 'trajectory', self.config.get('velocity_dt', .001))
        stress = predict_stress(deform, x, t)
        acceleration = temporal.derivative(lambda s: temporal.position(deform, x, s), t, self.config.get('acceleration_dt', 1/199), 2)
        r, valid = residual(now, v, stress, acceleration, gaussians.force_sigma, all_ids, first, anchors, n_first, n_anchor)
        if not valid.any(): raise ValueError('No valid material neighborhoods')
        raw = F.huber_loss(r[valid], torch.zeros_like(r[valid]), delta=1.)
        if not raw.isfinite(): raise FloatingPointError('Nonfinite shared momentum')
        ramp = min(1., (iteration-start)/max(1, self.config.get('momentum_ramp', 4000)))
        self.append('material_momentum.jsonl', {'iteration': iteration, 'raw_huber': float(raw.detach()),
            'residual_rms': float(r[valid].detach().square().mean().sqrt()), 'valid_fraction': float(valid.float().mean()),
            'nodes': len(all_ids), 'anchors': len(anchors), 'ramp': ramp, 'velocity_source': 'trajectory', 'stress_source': 'shared'})
        self.audit('momentum', raw, gaussians, deform, iteration)
        return raw*ramp

    def after_update(self, iteration, gaussians):
        super().after_update(iteration, gaussians)
        if self.enabled and self.stress_source == 'shared':
            if any(p.requires_grad or p.grad is not None for p in gaussians.velocity_net.parameters()):
                raise ValueError('Unused legacy physical network received a gradient')
