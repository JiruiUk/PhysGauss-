"""Same tested v1 objectives, activated on freshly bound stable cold-start rows."""
import torch
from variants.transfer_integrity.runtime import Tuning as Transfer
from variants.color_repair.runtime import Tuning as Color
from variants.integrity_tuning.model import coverage_objective
from .stage import schedule,legacy_iteration,verify_binding_boundary


class Tuning(Transfer):
    def initialize_data(self,source):
        super().initialize_data(source)
        start=int(self.config['transfer_start'])
        verify_binding_boundary(start,int(self.binding['source_iteration']),int(self.binding['point_count']),int(self.binding['point_count']))

    def geometry_loss(self,iteration,g):
        start=int(self.config['transfer_start'])
        verify_binding_boundary(start,int(self.binding['source_iteration']),len(g.get_xyz),int(self.binding['point_count']))
        return super().geometry_loss(legacy_iteration(iteration,start),g)

    def coverage_loss(self,iteration,view,package,g,warp):
        inherited=Color.coverage_loss(self,iteration,view,package,g,warp)
        if self.config.get('record_transfer_sampling',False):
            self.append('camera_sampling.jsonl',{'iteration':iteration,'frame':view.image_name,'time':float(view.fid)})
        if not self.transfer_enabled:return inherited
        ramp,due=schedule(iteration,int(self.config['transfer_start']),int(self.config.get('transfer_interval',10)),int(self.config.get('transfer_ramp',2000)))
        if not ramp:return inherited
        if view.image_name not in self.train_names:raise ValueError('Only true training observations may supervise coverage')
        observed=coverage_objective(package['alpha'],view.coverage_target.to(package['alpha']))
        added=float(self.config.get('observed_coverage_weight',0))*ramp*observed
        if due:
            geometry,terms=self.geometry_loss(iteration,g)
            added+=int(self.config.get('transfer_interval',10))*ramp*geometry
            record={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in terms.items()}
            record.update(iteration=iteration,activation=int(self.config['transfer_start']),ramp=ramp,
                          observed_coverage=float(observed.detach()),weighted_loss=float(added.detach()))
            self.append('transfer_loss.jsonl',record)
        if not torch.isfinite(added):raise ValueError('Nonfinite scratch-integrity loss')
        return added if inherited is None else inherited+added
