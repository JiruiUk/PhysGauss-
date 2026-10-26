"""Training-alpha supervision, weak affine neighborhoods and late coverage seeds."""
import math
from pathlib import Path
import torch
from variants.v3_iso_scratch.runtime import Tuning as Parent
from variants.metric_tuning.runtime import read_json, write_json, sha256
from variants.coverage_tuning.data import frame_paths, read_alpha
from variants.coverage_tuning.model import mask_loss, local_affine_loss, neighborhoods, interior
from variants.coverage_tuning.density import CoverageAccumulator


class Tuning(Parent):
    def __init__(self, config_path, run):
        super().__init__(config_path,run)
        self.mask_weight = float(self.config.get('mask_weight',0.))
        self.affine_weight = float(self.config.get('affine_weight',0.))
        self.density_enabled = self.enabled and self.config.get('coverage_density',False)
        self.coverage = None
        for w in (self.mask_weight,self.affine_weight):
            if not math.isfinite(w) or w < 0: raise ValueError('Invalid coverage/affine weight')
        if self.affine_weight and not self.config.get('single_object',False):
            raise ValueError('Multi-object training requires explicit object IDs; single-object Rubber Duck only')

    def configure_scene(self, scene, source):
        super().configure_scene(scene,source)
        if not self.enabled or not (self.mask_weight or self.affine_weight or self.density_enabled): return
        paths = frame_paths(source,'train')
        self.train_names = set(paths)
        self.train_order = {n:i for i,n in enumerate(sorted(paths))}
        for view in scene.getTrainCameras():
            if view.image_name not in paths: raise ValueError('Unexpected training view')
            view.coverage_target = read_alpha(paths[view.image_name],(view.image_width,view.image_height))
        self.target_ids = sorted(paths)
        write_json(self.run/'mask_supervision.json',{'split':'train','frame_ids':self.target_ids,
            'source':'original RGBA alpha; not SAM/motion mask','heldout_masks_in_loss':False})

    def coverage_loss(self, iteration, view, package, gaussians, warp):
        if not self.enabled or not (self.mask_weight or self.affine_weight or self.density_enabled): return None
        if view.image_name not in self.train_names: raise ValueError('Only training masks may enter loss')
        target = view.coverage_target.to(package['alpha'].device)
        result = None
        if self.mask_weight:
            raw = mask_loss(package['alpha'],target)
            result = self.mask_weight*raw
            if iteration == 1 or iteration % 100 == 0:
                self.append('coverage_loss.jsonl',{'iteration':iteration,'mask_l1':raw.item(),
                    'weighted_mask':result.item(),'frame':view.image_name})
                if package['alpha'].requires_grad:
                    def audit(grad):
                        if not torch.isfinite(grad).all(): raise ValueError('Nonfinite alpha gradient')
                        self.append('coverage_gradient.jsonl',{'iteration':iteration,'alpha_l1':grad.abs().sum().item()})
                    package['alpha'].register_hook(audit)
        interval = int(self.config.get('affine_interval',10))
        start = int(self.config.get('affine_start',3000))
        if self.affine_weight and iteration > start and iteration % interval == 0 and torch.is_tensor(warp):
            # Detach canonical basis AND canonical contribution to y: this term
            # regularizes the learned trajectory, not the canonical point cloud.
            from variants.kinematic_tuning.model import center_trajectory
            x = gaussians.get_xyz.detach()
            y = center_trajectory(x,warp)
            a,n,v = neighborhoods(x,iteration,count=int(self.config.get('affine_samples',128)),
                k=8,radius=float(self.config.get('affine_radius',.1)))
            raw = local_affine_loss(x,y,a,n,v)
            ramp = min(1.,(iteration-start)/max(1,self.config.get('affine_ramp',2000)))
            weighted = self.affine_weight*interval*ramp*raw
            result = weighted if result is None else result+weighted
            if not torch.isfinite(raw): raise ValueError('Nonfinite local affine loss')
            if iteration % 100 == 0:
                self.append('affine_loss.jsonl',{'iteration':iteration,'loss':raw.item(),
                    'weighted_loss':weighted.item(),'valid_edges':v.sum().item()})
        if self.density_enabled:
            self.observe_coverage(iteration,view,package,target,len(gaussians.get_xyz))
        return result

    @torch.no_grad()
    def observe_coverage(self,iteration,view,package,target,count):
        start = int(self.config.get('coverage_start',16000))
        end = int(self.config.get('coverage_until',36000))
        if not start <= iteration < end or iteration % int(self.config.get('coverage_sample_interval',20)): return
        if self.coverage is None:
            self.coverage = CoverageAccumulator(count,len(self.train_names),target.device)
        if len(self.coverage.score) != count: raise ValueError('Coverage point IDs changed inside window')
        core = interior(target)
        deficit = ((.95-package['alpha'].detach()).clamp_min(0)*core)
        deficit *= (package['alpha'].detach() < .7)
        # The CUDA renderer stores up to twenty actual footprint contributors.
        # Empty pixels have -1 IDs: never invent a depth or unobserved 3D point.
        stride = 4
        self.coverage.observe(self.train_order[view.image_name],
            package['gs_per_pixel'][:,::stride,::stride],deficit[:,::stride,::stride])

    @torch.no_grad()
    def after_update(self,iteration,gaussians):
        if not self.density_enabled: return
        interval = int(self.config.get('coverage_window',2000))
        start = int(self.config.get('coverage_start',16000))
        end = int(self.config.get('coverage_until',36000))
        if not start < iteration <= end or iteration % interval or self.coverage is None: return
        count = len(gaussians.get_xyz)
        limit = min(int(self.config.get('coverage_add_limit',512)),max(0,int(self.config.get('coverage_max_points',200000))-count))
        ids = self.coverage.select(int(self.config.get('coverage_min_views',3)),limit)
        added = len(ids)
        if added:
            # Conservative footprint-local seeds. Keep all parents; no late
            # opacity reset/prune. Dedicated RNG does not consume the trainer RNG.
            g = torch.Generator(device=gaussians.get_xyz.device).manual_seed(104729+iteration)
            sigma = gaussians.get_scaling[ids].clamp_max(.02)
            offset = torch.randn((added,3),device=sigma.device,generator=g)*sigma*.5
            opacity = gaussians.get_opacity[ids].clamp_max(.1).clamp_min(.01)
            gaussians.densification_postfix(gaussians.get_xyz[ids]+offset,
                gaussians._features_dc[ids],gaussians._features_rest[ids],torch.logit(opacity),
                gaussians._scaling[ids],gaussians._rotation[ids],
                new_indices=gaussians.particle_indices[ids],new_dynamic_mask=gaussians.dynamic_mask[ids])
        self.append('coverage_density.jsonl',{'iteration':iteration,'before':count,'added':added,
            'after':len(gaussians.get_xyz),'min_distinct_training_times':self.config.get('coverage_min_views',3),
            'parents_kept':True,'empty_pixels_have_no_3d_seeds':True})
        self.coverage.reset(len(gaussians.get_xyz))

    def save(self,checkpoint,manifest):
        super().save(checkpoint,manifest)
        if not self.density_enabled: return
        destination = Path(checkpoint).with_suffix('.coverage.pt')
        temporary = destination.with_suffix('.tmp')
        torch.save({'checkpoint_sha256':manifest['sha256'],'iteration':manifest['iteration'],
                    'target_ids':self.target_ids,'coverage':None if self.coverage is None else self.coverage.state_dict()},temporary)
        temporary.replace(destination)
        path = Path(checkpoint).with_suffix('.metrics.json'); state = read_json(path)
        state['coverage_sha256'] = sha256(destination); write_json(path,state)

    def restore(self,checkpoint,iteration):
        super().restore(checkpoint,iteration)
        if not self.density_enabled or iteration == 0: return
        path = Path(checkpoint).with_suffix('.coverage.pt')
        if sha256(path) != read_json(Path(checkpoint).with_suffix('.metrics.json'))['coverage_sha256']:
            raise ValueError('Coverage state checksum mismatch')
        state = torch.load(path,map_location='cpu',weights_only=True)
        if state['checkpoint_sha256'] != sha256(checkpoint) or state['iteration'] != iteration or state['target_ids'] != self.target_ids:
            raise ValueError('Coverage state / training frame mismatch')
        self.coverage = None if state['coverage'] is None else CoverageAccumulator.from_state(state['coverage'],'cuda')
