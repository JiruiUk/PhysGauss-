"""C1 plus optional fixed/learned shared effective-E constitutive consistency."""
from pathlib import Path
import math
import torch
from variants.color_repair.runtime import Tuning as Parent
from variants.kinematic_tuning.model import center_trajectory
from variants.metric_tuning.runtime import read_json, write_json, sha256
from pidg_ext.deformation_gradient import CanonicalGraph, estimate_F
from pidg_ext.losses import predicted_stress, constitutive_consistency
from .model import SharedYoung, conditional_profile


class Tuning(Parent):
    def __init__(self, config_path, run):
        super().__init__(config_path, run)
        self.material_active=self.enabled and self.config.get('material_enabled',False)
        self.material=None; self.material_optimizer=None; self.material_graph=None
        self.material_updates=0
        if not self.material_active:return
        if not self.config.get('ack_global_effective_material',False):
            raise ValueError('No true material labels: explicitly acknowledge a global effective prior')
        self.material=SharedYoung(self.config.get('young_init',1.),self.config.get('poisson_fixed',.3),
            self.config.get('learn_young',False),self.config.get('young_lower',.05),self.config.get('young_upper',20.))
        self.material_graph=CanonicalGraph(self.config.get('material_neighbors',16),
                                          self.config.get('material_graph_refresh',1000))
        for key in ('material_weight','material_prior_weight','material_lr'):
            value=float(self.config.get(key,1e-4))
            if not math.isfinite(value) or value<0:raise ValueError('Invalid '+key)
        if int(self.config.get('material_interval',100))<1:raise ValueError('Positive physics interval required')

    def after_scene(self, gaussians, deform):
        super().after_scene(gaussians,deform)
        if not self.material_active:return
        self.material.to(gaussians.get_xyz.device)
        if self.config.get('learn_young',False):
            self.material_optimizer=torch.optim.Adam(self.material.parameters(),lr=self.config.get('material_lr',1e-4))
        write_json(self.run/'material_scope.json',{'assumption':'one global effective material prior',
            'true_material_labels_available':False,'object_id_isolation':False,
            'units':'normalized effective stiffness; not calibrated Pa',
            'learned_scalar_count':1 if self.material_optimizer else 0,
            'poisson_fixed':self.material.poisson,'external_force_or_mass_recovery':False})

    def coverage_loss(self, iteration, view, package, gaussians, warp):
        parent=super().coverage_loss(iteration,view,package,gaussians,warp)
        if not self.material_active:return parent
        start=int(self.config.get('material_start',3000));interval=int(self.config.get('material_interval',100))
        if iteration<=start or iteration%interval:return parent
        if view.image_name not in self.train_names:raise ValueError('Constitutive queries must use a training time')
        X=gaussians.get_xyz.detach()
        x=center_trajectory(X,warp)
        graph=self.material_graph.get(X,iteration)
        count=min(len(X),int(self.config.get('material_samples',256)))
        ids=(torch.arange(count,device=X.device)*len(X)//count+iteration//interval)%len(X)
        estimate=estimate_F(X,x,graph,ids,min_determinant=self.config.get('material_min_J',1e-6))
        time=view.fid.detach().reshape(1,1).expand(len(ids),1).to(X)
        coords=torch.cat((X[ids],time),-1).requires_grad_(True)
        pred=predicted_stress(coords,gaussians.particle_indices[ids],gaussians.velocity_net,gaussians.force_sigma)
        raw, const=constitutive_consistency(pred,estimate['F'],self.material(),estimate['confidence'],estimate['valid'])
        ramp=min(1.,(iteration-start)/max(1,int(self.config.get('material_ramp',2000))))
        prior=self.material.prior()
        effective=self.config.get('material_weight',1e-4)*interval*ramp
        loss=effective*(raw+self.config.get('material_prior_weight',1e-3)*prior)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite material loss')
        grads=torch.autograd.grad(loss,(estimate['F'],pred),retain_graph=True,allow_unused=True)
        record={'iteration':iteration,'frame':view.image_name,'E':self.material.young.item(),
            'poisson':self.material.poisson,'learn_young':self.config.get('learn_young',False),
            'raw_constitutive':raw.item(),'weighted_loss':loss.item(),'prior_log_ratio_squared':prior.item(),
            'schedule_ramp':ramp,'interval_compensation':interval,
            'valid_F_fraction':estimate['valid'].float().mean().item(),'mean_det_F':estimate['J'].mean().item(),
            'F_gradient_norm':0. if grads[0] is None else grads[0].norm().item(),
            'predicted_stress_gradient_norm':0. if grads[1] is None else grads[1].norm().item(),
            'material_time':'same normalized fid as rendered deformation',
            'data_identifiability':'not_established','global_effective_material_assumption':True}
        if self.material.raw_E.requires_grad:
            g=torch.autograd.grad(loss,self.material.raw_E,retain_graph=True,allow_unused=True)[0]
            record['raw_E_gradient']=None if g is None else g.item()
        self.append('material_loss.jsonl',record)
        if iteration%int(self.config.get('material_profile_interval',1000))==0:
            profile=conditional_profile(pred,estimate['F'],self.material,estimate['confidence'],estimate['valid'])
            self.append('material_profile.jsonl',{'iteration':iteration,**profile})
        return loss if parent is None else parent+loss

    def after_update(self, iteration, gaussians):
        super().after_update(iteration,gaussians)
        if self.material_optimizer is None or self.material.raw_E.grad is None:return
        if not torch.isfinite(self.material.raw_E.grad):raise FloatingPointError('Nonfinite stiffness gradient')
        self.material_optimizer.step();self.material_optimizer.zero_grad(set_to_none=True)
        self.material_updates+=1

    def save(self, checkpoint, manifest):
        super().save(checkpoint,manifest)
        if not self.material_active:return
        from reproduction.scripts.training_checkpoint import cpu_tree
        path=Path(checkpoint).with_suffix('.material.pt');temporary=path.with_suffix('.tmp')
        torch.save(cpu_tree({'checkpoint_sha256':manifest['sha256'],'iteration':manifest['iteration'],
            'material':self.material.state_dict(),
            'optimizer':None if self.material_optimizer is None else self.material_optimizer.state_dict(),
            'updates':self.material_updates,'neighbors':self.material_graph.neighbors,
            'graph_iteration':self.material_graph.last_iteration}),temporary)
        temporary.replace(path)
        sidecar=Path(checkpoint).with_suffix('.metrics.json');state=read_json(sidecar)
        state['material_sha256']=sha256(path);write_json(sidecar,state)

    def restore(self, checkpoint, iteration):
        super().restore(checkpoint,iteration)
        if not self.material_active:return
        path=Path(checkpoint).with_suffix('.material.pt')
        if not path.exists():
            branch=read_json(self.run/'branch.json')
            if branch.get('initialize_material_at_branch') is not True or \
               branch['iteration']!=iteration or iteration!=self.config.get('material_start',3000) or \
               branch['identical_full_checkpoint_sha256']!=sha256(checkpoint):
                raise ValueError('Missing material checkpoint state')
            return
        state=read_json(Path(checkpoint).with_suffix('.metrics.json'))
        if state.get('material_sha256')!=sha256(path):raise ValueError('Material checkpoint checksum mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['iteration']!=iteration or saved['checkpoint_sha256']!=sha256(checkpoint):
            raise ValueError('Material checkpoint belongs to different full state')
        self.material.load_state_dict(saved['material'])
        if (saved['optimizer'] is None)!=(self.material_optimizer is None):raise ValueError('Material optimizer mode changed')
        if self.material_optimizer is not None:
            self.material_optimizer.load_state_dict(saved['optimizer'])
            for values in self.material_optimizer.state.values():
                for key,value in values.items():
                    if torch.is_tensor(value) and key!='step':values[key]=value.to(self.material.raw_E.device)
        self.material_updates=saved['updates']
        self.material_graph.neighbors=None if saved['neighbors'] is None else saved['neighbors'].to(self.material.raw_E.device)
        self.material_graph.last_iteration=saved['graph_iteration']
