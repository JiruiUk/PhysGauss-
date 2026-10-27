"""Same component formulas, explicit unsegmented scope, no mask loss."""
import math
import torch
from variants.v3_iso_scratch.runtime import Tuning as Scratch
from variants.coverage_tuning.runtime import Tuning as Coverage
from variants.coverage_tuning.model import local_affine_loss, neighborhoods
from variants.coverage_tuning.data import frame_paths
from variants.metric_tuning.runtime import write_json


class Tuning(Coverage):
    def __init__(self, config_path, run):
        # The old single_object declaration was disproved by scene inspection.
        # Keep the implemented loss unchanged; require honest acknowledgment of
        # its unsegmented scope rather than silently asserting object isolation.
        Scratch.__init__(self,config_path,run)
        self.mask_weight = float(self.config.get('mask_weight',0.))
        self.affine_weight = float(self.config.get('affine_weight',0.))
        self.density_enabled = self.enabled and self.config.get('coverage_density',False)
        self.coverage = None
        if self.mask_weight != 0: raise ValueError('Isolated arms must not include a mask loss')
        if not math.isfinite(self.affine_weight) or self.affine_weight<0: raise ValueError('Invalid affine weight')
        if self.affine_weight and self.density_enabled: raise ValueError('Enable only one isolated component')
        if self.affine_weight and not self.config.get('ack_unsegmented_affine',False):
            raise ValueError('Explicit acknowledgment of unsegmented neighbors is required')

    def configure_scene(self,scene,source):
        if self.density_enabled:
            # Density still needs training alpha as an addition-selection signal,
            # although its differentiable mask-loss coefficient is exactly zero.
            Coverage.configure_scene(self,scene,source)
            write_json(self.run/'component_supervision.json',{'mask_loss_weight':0.,
                'alpha_use':'training-only coverage candidate selection','object_isolation':False})
        else:
            Scratch.configure_scene(self,scene,source)
            if not self.enabled: return
            paths = frame_paths(source,'train')
            self.train_names = set(paths)
            if {v.image_name for v in scene.getTrainCameras()} != self.train_names:
                raise ValueError('Training camera membership changed')
            # No separate alpha supervision is read or attached. The baseline
            # RGB loader's RGBA compositing remains unchanged in every arm.
            write_json(self.run/'component_supervision.json',{'mask_loss_weight':0.,'alpha_target_read':False,
                'alpha_use':'none','object_isolation':False})

    def coverage_loss(self,iteration,view,package,gaussians,warp):
        if not self.enabled: return None
        if not self.affine_weight:
            return Coverage.coverage_loss(self,iteration,view,package,gaussians,warp)
        if view.image_name not in self.train_names: raise ValueError('Only training views may enter affine loss')
        interval = int(self.config.get('affine_interval',10))
        start = int(self.config.get('affine_start',3000))
        if iteration<=start or iteration%interval or not torch.is_tensor(warp): return None
        from variants.kinematic_tuning.model import center_trajectory
        x = gaussians.get_xyz.detach()
        y = center_trajectory(x,warp)
        a,n,v = neighborhoods(x,iteration,count=int(self.config.get('affine_samples',128)),
            k=8,radius=float(self.config.get('affine_radius',.1)))
        raw = local_affine_loss(x,y,a,n,v)
        ramp = min(1.,(iteration-start)/max(1,self.config.get('affine_ramp',2000)))
        weighted = self.affine_weight*interval*ramp*raw
        if not torch.isfinite(raw): raise ValueError('Nonfinite local affine loss')
        if iteration%100==0:
            self.append('affine_loss.jsonl',{'iteration':iteration,'loss':raw.item(),
                'weighted_loss':weighted.item(),'valid_edges':v.sum().item(),'object_isolation':False})
            if y.requires_grad:
                def audit(grad):
                    if not torch.isfinite(grad).all(): raise ValueError('Nonfinite affine trajectory gradient')
                    self.append('affine_gradient.jsonl',{'iteration':iteration,'trajectory_l1':grad.abs().sum().item()})
                y.register_hook(audit)
        return weighted
