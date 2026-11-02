"""Keep the C1 clock, independent velocity and original momentum path unchanged."""
from types import SimpleNamespace
import json
import numpy as np
import torch
from variants.color_repair.runtime import Tuning as Parent
from variants.flow_comparison import temporal
from variants.flow_comparison.geometry import flow_loss
from variants.kinematic_tuning.model import center_trajectory
from variants.unified_flow.flow import pixel_flow, next_projection
from variants.unified_flow.prepare_data import CACHE


class Tuning(Parent):
    def __init__(self, *args):
        super().__init__(*args)
        self.observed_flow_enabled = self.enabled and self.config.get('observed_flow_enabled', False)
        self.flow_tensors = {}
        self.flow_audited = False

    def configure_scene(self, scene, source):
        super().configure_scene(scene, source)
        if not self.observed_flow_enabled:
            return
        manifest = json.loads((CACHE/'manifest.json').read_text())
        self.flow_pairs = {p['source']: p for p in manifest['pairs']}
        self.train_cameras = {int(c.image_name): c for c in scene.getTrainCameras()}
        if set(self.train_cameras) != set(manifest['train_frame_ids']):
            raise ValueError('Optical-flow cache differs from C1 training membership')
        self.append('flow_contract.jsonl', {'target': 'raw source-to-target-camera UniMatch flow',
            'velocity_source': 'unchanged C1 independent velocity', 'estimated_flow_not_GT': True,
            'camera': 'actual loader FoV', 'metadata': 'negative x_mu; pre-update alpha*T',
            'no_teacher_depth': True, 'train_only': True})

    def flow_due(self, iteration):
        return (self.observed_flow_enabled and iteration > self.config['flow_start']
                and iteration % self.config['flow_interval'] == 0)

    def observed_loss(self, iteration, gaussians, deform, view, warp, render, pipe, background):
        if self.enabled and self.config.get('record_sampling', False):
            self.append('camera_sampling.jsonl', {'iteration': iteration, 'frame': view.image_name,
                'time': float(view.fid.item()), 'points': len(gaussians.get_xyz)})
        if not self.flow_due(iteration):
            return None
        pair = self.flow_pairs.get(int(view.image_name))
        if pair is None:
            self.append('observed_flow.jsonl', {'iteration': iteration, 'source': int(view.image_name),
                'skipped': 'no future training frame'})
            return None
        if not isinstance(warp, dict):
            raise ValueError('Optical flow requires an active C1 deformation')
        key = pair['source']
        if key not in self.flow_tensors:
            with np.load(CACHE/pair['file'], allow_pickle=False) as data:
                self.flow_tensors[key] = tuple(torch.from_numpy(data[k].copy()) for k in ('observed', 'valid'))
        target, observed_mask = [v.to(gaussians.get_xyz.device) for v in self.flow_tensors[key]]
        def low(camera):
            return SimpleNamespace(image_width=400, image_height=225, FoVx=camera.FoVx, FoVy=camera.FoVy,
                world_view_transform=camera.world_view_transform.to(background.device),
                full_proj_transform=camera.full_proj_transform.to(background.device),
                camera_center=camera.camera_center.to(background.device))
        current_camera, future_camera = low(view), low(self.train_cameras[pair['target']])
        with torch.no_grad():
            package = render(current_camera, gaussians, pipe, background, warp['d_xyz'].detach(),
                warp['d_rotation'].detach(), warp['d_scaling'].detach(), False,
                override_color=torch.zeros_like(gaussians.get_xyz))
        x = gaussians.get_xyz.detach()
        future = temporal.warp_at(deform, x, (pair['target']-1)/199)
        geometric, seen = pixel_flow(package, *next_projection(gaussians, future, future_camera), gaussians.get_opacity)
        t = float(view.fid.item())
        velocity = temporal.velocity(gaussians, deform, x, gaussians.particle_indices, t, 'independent')
        current = center_trajectory(gaussians.get_xyz, warp['d_xyz'])
        advected, _ = pixel_flow(package, *next_projection(gaussians, future, future_camera,
            current+pair['dt']*velocity), gaussians.get_opacity)
        valid = observed_mask & seen
        if not valid.any():
            raise ValueError('No valid optical-flow training coverage')
        geometry_loss = flow_loss(geometric, target, valid)
        velocity_loss = flow_loss(advected, target, valid)
        raw = .5*(geometry_loss+velocity_loss)
        if not bool(raw.isfinite()):
            raise FloatingPointError('Nonfinite C1 optical-flow objective')
        ramp = min(1., (iteration-self.config['flow_start'])/max(1, self.config['flow_ramp']))
        weighted = self.config['observed_flow_weight']*ramp*raw
        self.append('observed_flow.jsonl', {'iteration': iteration, 'source': pair['source'], 'target': pair['target'],
            'dt': pair['dt'], 'geometry_loss': float(geometry_loss.detach()),
            'velocity_loss': float(velocity_loss.detach()), 'weighted_loss': float(weighted.detach()),
            'ramp': ramp, 'valid_pixels': int(valid.sum()), 'observed_pixels': int(observed_mask.sum()),
            'velocity_source': 'independent'})
        if self.config.get('audit_flow_gradients', False) and not self.flow_audited:
            for name, module in {'grid': deform.grid4d, 'deformation': deform.deform,
                    'velocity_head': gaussians.velocity_net.velocity_head}.items():
                params = [p for p in module.parameters() if p.requires_grad]
                grads = torch.autograd.grad(weighted, params, retain_graph=True, allow_unused=True)
                magnitude = sum(float(g.detach().abs().sum()) for g in grads if g is not None)
                finite = all(bool(g.isfinite().all()) for g in grads if g is not None)
                if not finite or magnitude <= 0:
                    raise ValueError('Missing/nonfinite optical-flow gradient: '+name)
                self.append('flow_gradients.jsonl', {'iteration': iteration, 'group': name,
                    'gradient_l1': magnitude, 'finite': finite})
            self.flow_audited = True
        return weighted
