"""Independent unit-covariance and train-error sampling ablations on frozen H3."""
from pathlib import Path
import torch
from variants.integrity_tuning.runtime import Tuning as Parent
from variants.metric_tuning.model import quantize_png
from variants.metric_tuning.runtime import read_json,write_json
from .sampling import FrameSampler


class Tuning(Parent):
    def configure_scene(self,scene,source):
        super().configure_scene(scene,source)
        if not self.integrity_enabled:return
        self.frame_sampler=FrameSampler(self.train_names,self.config.get('integrity_hard_frame_probability',0.),
            self.config.get('integrity_hard_frame_fraction',.25))

    def select_training_camera(self,iteration,current,views):
        if not self.integrity_enabled or not self.config.get('integrity_hard_frame_probability',0.):return current
        return self.frame_sampler.select(current,views,enabled=iteration>=self.config.get('integrity_hard_frame_start',8000))

    def coverage_loss(self,iteration,view,package,gaussians,warp):
        result=super().coverage_loss(iteration,view,package,gaussians,warp)
        if not self.integrity_enabled or not self.config.get('integrity_hard_frame_probability',0.):return result
        with torch.no_grad():
            target=view.coverage_target.to(package['render']);fg=target>.5
            error=(package['render']-view.original_image[:3].to(package['render'])).abs().mean(0,keepdim=True)
            value=float((error*fg).sum()/fg.sum().clamp_min(1))
            self.frame_sampler.observe(view.image_name,value)
        if iteration%1000==0:
            self.append('frame_sampling.jsonl',{'iteration':iteration,'supervision':'train only',**self.frame_sampler.state()})
        return result

    def evaluate(self,iteration,scene,deform,render,pipe,background,on_the_fly):
        if not self.integrity_enabled:return super().evaluate(iteration,scene,deform,render,pipe,background,on_the_fly)
        rows=[]
        def tracked(view,*args,**kwargs):
            package=render(view,*args,**kwargs)
            a=quantize_png(package['render']).detach();b=quantize_png(view.original_image[:3].to(a))
            value=float(-10*torch.log10((a-b).square().mean().clamp_min(1e-12)))
            rows.append({'frame':view.image_name,'PSNR':value})
            return package
        result=super().evaluate(iteration,scene,deform,tracked,pipe,background,on_the_fly)
        rows=rows[:self.validation_count]
        self.append('temporal_validation.jsonl',{'iteration':iteration,'split':'validation','per_frame':rows,
            'worst5_PSNR':sum(sorted(r['PSNR'] for r in rows)[:5])/5})
        return result

    def save(self,checkpoint,manifest):
        super().save(checkpoint,manifest)
        if not self.integrity_enabled or not self.config.get('integrity_hard_frame_probability',0.):return
        path=Path(checkpoint).with_suffix('.metrics.json');state=read_json(path)
        state['frame_sampler']=self.frame_sampler.state();write_json(path,state)

    def restore(self,checkpoint,iteration):
        super().restore(checkpoint,iteration)
        if not self.integrity_enabled or not self.config.get('integrity_hard_frame_probability',0.):return
        state=read_json(Path(checkpoint).with_suffix('.metrics.json')).get('frame_sampler')
        if state is not None:self.frame_sampler.restore(state)
        elif iteration>3000 and not self.config.get('engineering_smoke'):raise ValueError('Missing trained sampler state')
