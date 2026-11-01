"""Optional model regularizers; no camera selection, data weighting or renderer edits."""
import math
import torch
from variants.integrity_tuning.runtime import Tuning as Parent,unregularized
from variants.integrity_tuning.model import persistent_edges
from variants.coverage_tuning.model import neighborhoods,local_affine_loss
from variants.color_repair.model import positive_color
from variants.kinematic_tuning.model import center_trajectory
from variants.metric_tuning.model import quantize_png
from .model import motion_gate,rendered_covariance,covariance_curvature,time_stencil


class Tuning(Parent):
    def __init__(self,config_path,run):
        super().__init__(config_path,run)
        for key in ['integrity_motion_affine_weight','integrity_covariance_weight']:
            if not math.isfinite(self.config.get(key,0.)) or self.config.get(key,0.)<0:raise ValueError('Invalid model regularization weight')

    def coverage_loss(self,iteration,view,package,gaussians,warp):
        result=super().coverage_loss(iteration,view,package,gaussians,warp)
        wa=self.config.get('integrity_motion_affine_weight',0.);wc=self.config.get('integrity_covariance_weight',0.)
        if not self.integrity_enabled or not (wa or wc):return result
        start=self.config.get('integrity_model_start',8000);interval=self.config.get('integrity_model_interval',20)
        if iteration<=start or iteration%interval:return result
        if view.image_name not in self.train_names:raise ValueError('Only the unchanged training camera may enter these losses')
        x=gaussians.get_xyz.detach();deform=self.integrity_deform;ramp=min(1.,(iteration-start)/self.config.get('integrity_ramp',4000))
        loss=package['alpha'].sum()*0;record={'iteration':iteration,'frame':view.image_name,'split':'train','sampler_unchanged':True}
        if wa:
            a,n,valid=neighborhoods(x,iteration,count=self.config.get('integrity_model_anchors',96),k=8,radius=.05)
            opacity=gaussians.get_opacity.detach().reshape(-1)
            color=positive_color(.28209479177387814*gaussians._features_dc.detach()[:,0]+.5).clamp(0,1)
            valid&=(opacity[a,None]>.05)&(opacity[n]>.05)&((color[n]-color[a,None]).norm(dim=-1)<.25)
            ids=torch.unique(torch.cat((a,n.flatten())));al,nl=torch.searchsorted(ids,a),torch.searchsorted(ids,n)
            with unregularized(deform),torch.no_grad():
                tracks=[]
                for t in (self.integrity_times[0],self.integrity_times[len(self.integrity_times)//2],self.integrity_times[-1]):
                    p=deform.step(x[ids],x.new_full((len(ids),1),t))
                    tracks.append(center_trajectory(x[ids],p['d_xyz']))
                before=int(valid.sum());valid=persistent_edges(tracks,al,nl,valid,max_distance=.15)
                valid=motion_gate(x[ids],tracks,al,nl,valid)
            raw=local_affine_loss(x,center_trajectory(x,warp),a,n,valid)
            loss=loss+wa*interval*ramp*raw
            record.update(motion_affine=float(raw.detach()),candidate_edges=before,valid_motion_edges=int(valid.sum()),
                usable_anchors=int((valid.sum(-1)>=4).sum()),object_instance_labels_used=False)
        if wc:
            pool=torch.nonzero(package['visibility_filter']&(gaussians.get_opacity.detach().reshape(-1)>.05)).reshape(-1)
            if len(pool):
                count=min(len(pool),self.config.get('integrity_covariance_samples',512))
                ids=pool[(torch.arange(count,device=x.device)*len(pool)//count+iteration//interval)%len(pool)]
                scale=gaussians.get_scaling.detach()[ids];rotation=gaussians.get_rotation.detach()[ids]
                times=time_stencil(float(view.fid.item()),self.config.get('integrity_covariance_h',1/199))
                cov=[]
                with unregularized(deform):
                    for t in times:
                        p=deform.step(x[ids],x.new_full((len(ids),1),t))
                        cov.append(rendered_covariance(scale+p['d_scaling'],rotation+p['d_rotation']))
                raw,stats=covariance_curvature(*cov,
                    shape_tolerance=self.config.get('integrity_covariance_shape_tolerance',.005),
                    size_tolerance=self.config.get('integrity_covariance_size_tolerance',.05))
                loss=loss+wc*interval*ramp*raw
                record.update(covariance_curvature=float(raw.detach()),stencil=list(times),samples=count,**{k:float(v) for k,v in stats.items()})
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite model regularization')
        if iteration%100==0:self.append('model_regularization.jsonl',{**record,'weighted_loss':float(loss.detach())})
        return loss if result is None else result+loss

    def evaluate(self,iteration,scene,deform,render,pipe,background,on_the_fly):
        if not self.integrity_enabled:return super().evaluate(iteration,scene,deform,render,pipe,background,on_the_fly)
        rows=[]
        def tracked(view,*args,**kwargs):
            package=render(view,*args,**kwargs)
            a=quantize_png(package['render']).detach();b=quantize_png(view.original_image[:3].to(a))
            rows.append({'frame':view.image_name,'PSNR':float(-10*torch.log10((a-b).square().mean().clamp_min(1e-12)))})
            return package
        value=super().evaluate(iteration,scene,deform,tracked,pipe,background,on_the_fly)
        rows=rows[:self.validation_count]
        self.append('temporal_validation.jsonl',{'iteration':iteration,'split':'validation','per_frame':rows,
            'worst5_PSNR':sum(sorted(r['PSNR'] for r in rows)[:5])/5})
        return value
