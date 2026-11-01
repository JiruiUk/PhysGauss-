"""Optional hard-foreground RGB term on the frozen integrity runtime."""
import torch
from variants.integrity_tuning.runtime import Tuning as Parent
from .model import foreground_tail_loss


class Tuning(Parent):
    def coverage_loss(self,iteration,view,package,gaussians,warp):
        result=super().coverage_loss(iteration,view,package,gaussians,warp)
        weight=self.config.get('integrity_tail_weight',0.)
        start=self.config.get('integrity_tail_start',8000);interval=self.config.get('integrity_tail_interval',4)
        if not self.integrity_enabled or not weight or iteration<=start or iteration%interval:return result
        if view.image_name not in self.train_names:raise ValueError('RGB-tail term requires training frames')
        covered_target=view.coverage_target.to(package['render'])*(package['alpha'].detach()>=self.config.get('integrity_tail_alpha_min',.8))
        raw=foreground_tail_loss(package['render'],view.original_image[:3].to(package['render']),
            covered_target,self.config.get('integrity_tail_fraction',.2))
        value=weight*interval*min(1.,(iteration-start)/self.config.get('integrity_ramp',4000))*raw
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite foreground-tail loss')
        if iteration%100==0:
            record={'iteration':iteration,'frame':view.image_name,'raw':float(raw.detach()),'weighted_loss':float(value.detach()),'split':'train'}
            if self.config.get('integrity_tail_audit',False):
                gradient=torch.autograd.grad(raw,gaussians._features_dc,retain_graph=True,allow_unused=True)[0]
                record['SH_gradient_norm']=0. if gradient is None else float(gradient.norm())
            self.append('foreground_tail.jsonl',record)
        return value if result is None else result+value
