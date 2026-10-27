"""Repair package: stable color, staged mask/foreground RGB, weak affine, color-aware clones."""
import math
from pathlib import Path
import torch
from variants.v3_iso_scratch.runtime import Tuning as Scratch
from variants.coverage_tuning.runtime import Tuning as Coverage
from variants.coverage_tuning.model import mask_loss,local_affine_loss,neighborhoods,interior
from variants.coverage_tuning.density import CoverageAccumulator
from variants.coverage_tuning.data import frame_paths,read_alpha
from variants.color_repair.model import inverse_positive_color,foreground_rgb_loss
from variants.color_repair.evidence import ColorEvidence
from variants.metric_tuning.runtime import read_json,write_json,sha256


class Tuning(Coverage):
    def __init__(self,config_path,run):
        Scratch.__init__(self,config_path,run)
        self.mask_weight=float(self.config.get('mask_weight',0.))
        self.affine_weight=float(self.config.get('affine_weight',0.))
        self.fg_weight=float(self.config.get('foreground_rgb_weight',0.))
        self.density_enabled=self.enabled and self.config.get('coverage_density',False)
        self.coverage=None;self.color_evidence=None
        for value in (self.mask_weight,self.affine_weight,self.fg_weight):
            if not math.isfinite(value) or value<0:raise ValueError('Invalid repair loss weight')
        if self.affine_weight and not self.config.get('ack_unsegmented_affine',False):
            raise ValueError('Real object-ID isolation is unavailable; acknowledge unsegmented neighbors')
        beta=float(self.config.get('color_beta',5.))
        if not math.isfinite(beta) or beta<=0:raise ValueError('Invalid positive-color beta')

    def after_scene(self,gaussians,deform):
        super().after_scene(gaussians,deform)
        if not self.enabled or not self.config.get('positive_color',False):return
        if self.warm_start:raise ValueError('Repair training must start from random initialization')
        if (self.run/'checkpoints/resume.pt').exists():return
        # Match the initial positive RGB despite changing the color activation.
        with torch.no_grad():
            old=.28209479177387814*gaussians._features_dc+.5
            color=old.clamp_min(1e-4)
            if torch.count_nonzero(gaussians._features_rest):raise ValueError('Expected random step0 SH with zero higher orders')
            raw=inverse_positive_color(color,float(self.config.get('color_beta',5.)))
            gaussians._features_dc.copy_((raw-.5)/.28209479177387814)
            write_json(self.run/'initial_color_conversion.json',{'activation':'softplus','beta':self.config.get('color_beta',5.),
                'only_random_initialization':True,'maximum_initial_rgb_floor_change':float((old-color).abs().max()),
                'black_floor':1e-4,'not_a_pretrained_checkpoint':True})

    def configure_scene(self,scene,source):
        Scratch.configure_scene(self,scene,source)
        if self.enabled and (self.mask_weight or self.affine_weight or self.fg_weight or self.density_enabled):
            paths=frame_paths(source,'train');self.train_names=set(paths)
            self.train_order={name:i for i,name in enumerate(sorted(paths))};self.target_ids=sorted(paths)
            views=scene.getTrainCameras()
            if {view.image_name for view in views}!=self.train_names:raise ValueError('Training target membership mismatch')
            for view in views:
                view.coverage_target=read_alpha(paths[view.image_name],(view.image_width,view.image_height))
            write_json(self.run/'repair_scope.json',{'supervision':'120 training RGBA/RGB only',
                'object_identity_isolation':False,'foreground_rgb_weight':self.fg_weight,
                'smooth_color':self.config.get('positive_color',False),'preserves_V3_image_loss_and_time':True})

    def ramp(self,iteration):
        start=int(self.config.get('mask_start',3000));length=max(1,int(self.config.get('mask_ramp',2000)))
        return min(1.,max(0.,(iteration-start)/length))

    def coverage_loss(self,iteration,view,package,gaussians,warp):
        if not self.enabled or not (self.mask_weight or self.affine_weight or self.fg_weight or self.density_enabled):return None
        if view.image_name not in self.train_names:raise ValueError('Only training views may enter repair losses')
        target=view.coverage_target.to(package['alpha'].device)
        loss=None;ramp=self.ramp(iteration)
        if ramp and (self.mask_weight or self.fg_weight):
            mask=mask_loss(package['alpha'],target) if self.mask_weight else package['alpha'].sum()*0
            fg=foreground_rgb_loss(package['render'],view.original_image[:3].to(package['render']),target) if self.fg_weight else mask*0
            loss=ramp*(self.mask_weight*mask+self.fg_weight*fg)
            if iteration==int(self.config.get('mask_start',3000))+1 or iteration%100==0:
                self.append('repair_loss.jsonl',{'iteration':iteration,'frame':view.image_name,'ramp':ramp,
                    'mask_l1':mask.item(),'foreground_rgb_l1':fg.item(),'weighted_loss':loss.item()})
        interval=int(self.config.get('affine_interval',10));start=int(self.config.get('affine_start',3000))
        if self.affine_weight and iteration>start and iteration%interval==0 and torch.is_tensor(warp):
            from variants.kinematic_tuning.model import center_trajectory
            x=gaussians.get_xyz.detach();y=center_trajectory(x,warp)
            a,n,v=neighborhoods(x,iteration,count=int(self.config.get('affine_samples',128)),k=8,
                radius=float(self.config.get('affine_radius',.05)))
            raw=local_affine_loss(x,y,a,n,v)
            value=self.affine_weight*interval*min(1.,(iteration-start)/max(1,self.config.get('affine_ramp',2000)))*raw
            loss=value if loss is None else loss+value
            if iteration%100==0:self.append('affine_loss.jsonl',{'iteration':iteration,'loss':raw.item(),
                'weighted_loss':value.item(),'valid_edges':v.sum().item(),'object_isolation':False})
        if self.density_enabled:self.observe_coverage(iteration,view,package,target,len(gaussians.get_xyz))
        if loss is not None and not torch.isfinite(loss):raise ValueError('Nonfinite repaired loss')
        return loss

    @torch.no_grad()
    def observe_coverage(self,iteration,view,package,target,count):
        if not int(self.config.get('coverage_start',16000))<=iteration<int(self.config.get('coverage_until',36000)):
            return
        if iteration%int(self.config.get('coverage_sample_interval',20)):return
        if self.coverage is None:
            self.coverage=CoverageAccumulator(count,len(self.train_names),target.device)
            self.color_evidence=ColorEvidence(count,target.device)
        if len(self.coverage.score)!=count or len(self.color_evidence.mass)!=count:
            raise ValueError('Point IDs changed inside the color/coverage window')
        deficit=(.95-package['alpha'].detach()).clamp_min(0)*interior(target)*(package['alpha'].detach()<.7)
        ids=package['gs_per_pixel'][:,::4,::4];deficit=deficit[:,::4,::4]
        self.coverage.observe(self.train_order[view.image_name],ids,deficit)
        self.color_evidence.observe(ids,deficit,view.original_image[:3].to(target.device)[:,::4,::4])

    @torch.no_grad()
    def after_update(self,iteration,gaussians):
        if not self.density_enabled:return
        start=int(self.config.get('coverage_start',16000));end=int(self.config.get('coverage_until',36000))
        if not start<iteration<=end or iteration%int(self.config.get('coverage_window',2000)) or self.coverage is None:return
        count=len(gaussians.get_xyz)
        limit=min(int(self.config.get('coverage_add_limit',256)),max(0,int(self.config.get('coverage_max_points',200000))-count))
        ids=self.coverage.select(int(self.config.get('coverage_min_views',3)),count)
        mean,std=self.color_evidence.mean_std(ids)
        usable=(self.color_evidence.mass[ids]>1e-6)&(std.max(-1).values<=float(self.config.get('clone_color_std_max',.15)))
        ids=ids[usable][:limit];mean=mean[usable][:limit]
        added=len(ids)
        if added:
            generator=torch.Generator(device=gaussians.get_xyz.device).manual_seed(104729+iteration)
            offset=torch.randn((added,3),device=ids.device,generator=generator)*gaussians.get_scaling[ids].clamp_max(.02)*.25
            dc=(inverse_positive_color(mean.clamp(1e-4,.999),float(self.config.get('color_beta',5.)))-.5)/.28209479177387814
            opacity=gaussians.get_opacity[ids].clamp(.005,.05)
            gaussians.densification_postfix(gaussians.get_xyz[ids]+offset,dc[:,None,:],
                torch.zeros_like(gaussians._features_rest[ids]),torch.logit(opacity),
                gaussians._scaling[ids],gaussians._rotation[ids],
                new_indices=gaussians.particle_indices[ids],new_dynamic_mask=gaussians.dynamic_mask[ids])
        self.append('coverage_density.jsonl',{'iteration':iteration,'before':count,'added':added,'after':len(gaussians.get_xyz),
            'color_consistency_gate':True,'new_color_source':'weighted training RGB at deficit footprint pixels',
            'higher_order_SH_reset':True,'parents_kept':True,'empty_pixels_have_no_3d_seeds':True})
        self.coverage.reset(len(gaussians.get_xyz));self.color_evidence=ColorEvidence(len(gaussians.get_xyz),gaussians.get_xyz.device)

    def save(self,checkpoint,manifest):
        Coverage.save(self,checkpoint,manifest)
        if not self.density_enabled:return
        path=Path(checkpoint).with_suffix('.color.pt');temporary=path.with_suffix('.tmp')
        torch.save({'checkpoint_sha256':manifest['sha256'],'iteration':manifest['iteration'],
                    'evidence':None if self.color_evidence is None else self.color_evidence.state_dict()},temporary)
        temporary.replace(path)
        sidecar=Path(checkpoint).with_suffix('.metrics.json');state=read_json(sidecar)
        state['color_evidence_sha256']=sha256(path);write_json(sidecar,state)

    def restore(self,checkpoint,iteration):
        Coverage.restore(self,checkpoint,iteration)
        if not self.density_enabled or iteration==0:return
        path=Path(checkpoint).with_suffix('.color.pt')
        if sha256(path)!=read_json(Path(checkpoint).with_suffix('.metrics.json'))['color_evidence_sha256']:
            raise ValueError('Color-evidence checksum mismatch')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['iteration']!=iteration or state['checkpoint_sha256']!=sha256(checkpoint):raise ValueError('Wrong color-evidence state')
        self.color_evidence=None if state['evidence'] is None else ColorEvidence.from_state(state['evidence'],'cuda')
