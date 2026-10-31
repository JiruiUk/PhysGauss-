"""Opt-in completeness losses; C1 flags-off training/inference remains intact."""
import math
import json
from pathlib import Path
from contextlib import contextmanager
import torch
from variants.color_repair.runtime import Tuning as Parent
from variants.color_repair.model import positive_color
from variants.coverage_tuning.model import neighborhoods
from variants.coverage_tuning.data import frame_paths,read_alpha
from variants.kinematic_tuning.model import center_trajectory
from variants.metric_tuning.model import quantize_png
from variants.metric_tuning.runtime import read_json,write_json
from .model import coverage_objective,strain_objective,persistent_edges,image_metrics,aggregate


@contextmanager
def unregularized(deform):
    old=(deform.reg_spatial_able,deform.reg_temporal_able)
    deform.reg_spatial_able=deform.reg_temporal_able=False
    try:yield
    finally:deform.reg_spatial_able,deform.reg_temporal_able=old


def quality_guard(metrics,reference):
    return (metrics['PSNR']>=reference['PSNR']-.5 and metrics['SSIM']>=reference['SSIM']-.001
        and metrics['LPIPS']<=reference['LPIPS']+.003 and metrics['MS-SSIM']>=reference['MS-SSIM']-.001)


class Tuning(Parent):
    def __init__(self,config_path,run):
        super().__init__(config_path,run)
        self.integrity_enabled=self.enabled and self.config.get('integrity_enabled',False)
        self.integrity_best=None
        for key in ('integrity_coverage_weight','integrity_strain_weight'):
            value=self.config.get(key,0.)
            if not math.isfinite(value) or value<0:raise ValueError('Invalid integrity weight')

    def after_scene(self,gaussians,deform):
        super().after_scene(gaussians,deform);self.integrity_deform=deform

    def configure_scene(self,scene,source):
        super().configure_scene(scene,source)
        if not self.integrity_enabled:return
        paths=frame_paths(source,'test') # The held-out loader slot is validation30.
        if set(paths)&self.train_names:raise ValueError('Integrity validation/train overlap')
        for view in scene.getTestCameras():
            view.integrity_validation_target=read_alpha(paths[view.image_name],(view.image_width,view.image_height))
        self.integrity_times=sorted(float(v.fid.item()) for v in scene.getTrainCameras())
        if len(set(self.integrity_times))!=120:raise ValueError('Expected120 unique training timestamps')

    def coverage_loss(self,iteration,view,package,gaussians,warp):
        result=super().coverage_loss(iteration,view,package,gaussians,warp)
        if not self.integrity_enabled:return result
        if view.image_name not in self.train_names:raise ValueError('Integrity losses require training frames')
        added=package['alpha'].sum()*0;record={'iteration':iteration,'frame':view.image_name}
        weight=self.config.get('integrity_coverage_weight',0.)
        ramp=min(1.,max(0.,(iteration-self.config.get('integrity_coverage_start',3000))/self.config.get('integrity_ramp',4000)))
        if weight and ramp:
            raw=coverage_objective(package['alpha'],view.coverage_target.to(package['alpha']))
            added=added+weight*ramp*raw;record['balanced_coverage']=float(raw.detach())
        interval=self.config.get('integrity_strain_interval',20)
        weight=self.config.get('integrity_strain_weight',0.)
        ramp=min(1.,max(0.,(iteration-self.config.get('integrity_strain_start',8000))/self.config.get('integrity_ramp',4000)))
        if weight and ramp and iteration%interval==0:
            x=gaussians.get_xyz.detach();a,n,valid=neighborhoods(x,iteration,count=96,k=8,radius=.05)
            opacity=gaussians.get_opacity.detach().reshape(-1)
            color=positive_color(.28209479177387814*gaussians._features_dc.detach()[:,0]+.5).clamp(0,1)
            valid &= (opacity[a,None]>.05)&(opacity[n]>.05)&((color[n]-color[a,None]).norm(dim=-1)<.25)
            ids=torch.unique(torch.cat((a,n.flatten())))
            al,nl=torch.searchsorted(ids,a),torch.searchsorted(ids,n)
            current=float(view.fid.item());index=min(range(len(self.integrity_times)),key=lambda i:abs(self.integrity_times[i]-current))
            next_time=self.integrity_times[index+1 if index+1<len(self.integrity_times) else index-1]
            deform=self.integrity_deform
            with unregularized(deform):
                with torch.no_grad():
                    positions=[]
                    for t in (self.integrity_times[0],self.integrity_times[len(self.integrity_times)//2],self.integrity_times[-1]):
                        q=deform.step(x[ids],x.new_full((len(ids),1),t))
                        positions.append(center_trajectory(x[ids],q['d_xyz']))
                    valid=persistent_edges(positions,al,nl,valid,max_distance=.15)
                later=deform.step(x[ids],x.new_full((len(ids),1),next_time))
            now=center_trajectory(x[ids],warp[ids]);future=center_trajectory(x[ids],later['d_xyz'])
            tolerance=min(.2,.03*abs(next_time-current)*199)
            raw=strain_objective(now,future,al,nl,valid,tolerance=tolerance)
            added=added+weight*interval*ramp*raw
            record.update(temporal_strain=float(raw.detach()),valid_edges=int(valid.sum()),
                adjacent_training_time=next_time,tolerance=tolerance,instance_ids_used=False)
        if not torch.isfinite(added):raise FloatingPointError('Nonfinite integrity loss')
        if iteration%100==0:
            record['weighted_loss']=float(added.detach());self.append('integrity_loss.jsonl',record)
        return added if result is None else result+added

    def evaluate(self,iteration,scene,deform,render,pipe,background,on_the_fly):
        if not self.integrity_enabled:return super().evaluate(iteration,scene,deform,render,pipe,background,on_the_fly)
        rows=[]
        def tracked(view,*args,**kwargs):
            package=render(view,*args,**kwargs)
            alpha=quantize_png(package['alpha']).detach().cpu().numpy()
            image=quantize_png(package['render']).detach().permute(1,2,0).cpu().numpy()
            gt=quantize_png(view.original_image[:3]).detach().permute(1,2,0).cpu().numpy()
            rows.append({'frame':view.image_name,**image_metrics(alpha,view.integrity_validation_target.numpy(),image,gt)})
            return package
        value=super().evaluate(iteration,scene,deform,tracked,pipe,background,on_the_fly)
        # Parent diagnostics may render extra views; only the first30 are val.
        rows=rows[:self.validation_count]
        if len(rows)!=30:raise ValueError('Incomplete integrity validation')
        summary=aggregate(rows)
        current=json.loads((self.run/'metric_history.jsonl').read_text(encoding='utf-8').splitlines()[-1])['metrics']
        guarded=quality_guard(current,self.config['integrity_baseline_metrics']) and \
            summary['background_false_coverage']<=self.config['integrity_baseline_diagnostics']['background_false_coverage']+.002
        event={'iteration':iteration,'split':'validation','metrics':current,'integrity':summary,'per_frame':rows,'quality_guard':guarded}
        self.append('integrity_history.jsonl',event)
        if iteration==self.config.get('integrity_screen_iteration',20000):
            tag='screen'+str(iteration);scene.save(tag);deform.save_weights(self.run,tag)
        if guarded and (self.integrity_best is None or summary['integrity_score']<self.integrity_best['score']):
            self.integrity_best={'score':summary['integrity_score'],'iteration':iteration,'metrics':current,'integrity':summary,
                'snapshot':'integrity_best','selection_split':'validation'}
            scene.save('integrity_best');deform.save_weights(self.run,'integrity_best')
            write_json(self.run/'integrity_best.json',self.integrity_best)
        return value

    def save(self,checkpoint,manifest):
        super().save(checkpoint,manifest)
        if not self.integrity_enabled:return
        path=Path(checkpoint).with_suffix('.metrics.json')
        state=read_json(path);state['integrity_best']=self.integrity_best;write_json(path,state)

    def restore(self,checkpoint,iteration):
        super().restore(checkpoint,iteration)
        if not self.integrity_enabled:return
        self.integrity_best=read_json(Path(checkpoint).with_suffix('.metrics.json')).get('integrity_best')
