"""Training/inference hooks; flags-off retains the frozen repaired C1 path."""
import json
import os
from pathlib import Path
import re
import torch
from variants.color_repair.runtime import Tuning as Parent
from variants.metric_tuning.runtime import read_json,write_json,sha256
from .model import ObjectMotion


def configure_deformation(deform):
    path=os.environ.get('PIDG_METRIC_CONFIG')
    config=read_json(path) if path else {}
    deform.object_model=None;deform.object_cache=None;deform.object_iteration=0
    deform.object_config=config
    if not config.get('enabled') or not config.get('object_id_enabled',False):return
    device=deform.scale_xyz.device
    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(1729)
        deform.object_model=ObjectMotion(config.get('object_capacity',16),config.get('learn_object_identity',True)).to(device)


def blend(deform):
    c=deform.object_config
    return c.get('object_blend',.25)*min(1.,max(0.,(deform.object_iteration-c.get('object_start',3000))/max(1,c.get('object_ramp',5000))))


def apply_warp(deform,xyz,time,warp):
    if deform.object_model is None:return warp
    ratio=blend(deform)
    if ratio==0:return warp
    deform.object_model.initialize(xyz.detach())
    result,state=deform.object_model(xyz,time,warp,ratio)
    # The first query is the primary image trajectory. Optical-flow queries
    # later in the iteration must not replace its regularization state.
    if torch.is_grad_enabled() and deform.object_cache is None:deform.object_cache=state
    return result


def snapshot_save(deform,directory):
    if deform.object_model is None:return
    torch.save({'iteration':deform.object_iteration,'model':deform.object_model.state_dict()},Path(directory)/'object_id.pt')


def snapshot_load(deform,directory,run,tag):
    if deform.object_model is None:return
    path=Path(directory)/'object_id.pt'
    if path.exists():
        state=torch.load(path,map_location=deform.scale_xyz.device,weights_only=True)
        deform.object_model.load_state_dict(state['model']);deform.object_iteration=state['iteration'];return
    # Pre-object prefix snapshots are exact C1 and must keep blend zero.
    step=int(tag) if str(tag).isdigit() else None
    marker=Path(directory)/'iter.txt'
    if step is None and marker.exists():step=int(re.findall(r'\d+',marker.read_text())[-1])
    if step is None:
        best=read_json(Path(run)/'metric_best.json')
        steps=[v['iteration'] for v in best.values() if v['snapshot']==str(tag)]
        if steps:step=steps[0]
    if step is None or step>deform.object_config.get('object_start',3000):
        raise ValueError('Missing trained object state for selected snapshot')
    deform.object_iteration=step


class Tuning(Parent):
    def after_scene(self,gaussians,deform):
        super().after_scene(gaussians,deform)
        self.object_deform=deform;self.object_optimizer=None
        self.object_active=deform.object_model is not None
        if self.object_active:
            self.object_optimizer=torch.optim.Adam([p for p in deform.object_model.parameters() if p.requires_grad],
                                                   lr=self.config.get('object_lr',2e-4))

    def before_iteration(self,iteration):
        self.object_deform.object_iteration=iteration;self.object_deform.object_cache=None

    def coverage_loss(self,iteration,view,package,gaussians,warp):
        parent=super().coverage_loss(iteration,view,package,gaussians,warp)
        if not self.object_active or blend(self.object_deform)==0:return parent
        if view.image_name not in self.train_names:raise ValueError('Object loss must use train frames only')
        state=self.object_deform.object_cache
        if state is None:raise ValueError('Object motion not connected to primary image render')
        ramp=min(1.,max(0.,(iteration-self.config.get('object_start',3000))/max(1,self.config.get('object_ramp',5000))))
        loss,stats=self.object_deform.object_model.regularizer(state,
            self.config.get('object_alignment',1e-3),self.config.get('object_sharpness',5e-5),self.config.get('object_complexity',1e-6))
        loss=loss*ramp
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite object loss')
        interval=self.config.get('object_audit_interval',1000)
        if iteration%interval==0 or iteration==self.config.get('object_start',3000)+1:
            record={'iteration':iteration,'frame':view.image_name,'blend':blend(self.object_deform),
                'learn_identity':self.config.get('learn_object_identity',True),'capacity':self.config.get('object_capacity',16),
                'weighted_loss':float(loss.detach()),'instance_labels_used':False,'ID_accuracy_available':False,
                **{key:float(value) for key,value in stats.items()}}
            if self.config.get('learn_object_identity',True):
                image_loss=(package['render']-view.original_image[:3].to(package['render'])).square().mean()
                g=torch.autograd.grad(image_loss,self.object_deform.object_model.identity[-1].weight,
                                      retain_graph=True,allow_unused=True)[0]
                record['RGB_to_identity_gradient_norm']=0. if g is None else float(g.norm())
            self.append('object_loss.jsonl',record)
        return loss if parent is None else parent+loss

    def after_update(self,iteration,gaussians):
        super().after_update(iteration,gaussians)
        if not self.object_active:return
        model=self.object_deform.object_model
        if iteration<self.config.get('object_total_iterations',50000):
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite object gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            self.object_optimizer.step()
        self.object_optimizer.zero_grad(set_to_none=True);self.object_deform.object_cache=None
        if iteration>=self.config.get('object_prune_start',10000) and iteration%self.config.get('object_prune_interval',2000)==0:
            with torch.no_grad():
                x=gaussians.get_xyz.detach()[::max(1,len(gaussians.get_xyz)//4096)][:4096]
                if bool(model.initialized):
                    mass=model.prune_unused(x,self.config.get('object_min_mass',.001))
                    self.append('object_prune.jsonl',{'iteration':iteration,'active':int(model.active.sum()),
                        'candidate_capacity':model.capacity,'mass':mass.tolist(),'true_object_count_available':False})

    def save(self,checkpoint,manifest):
        super().save(checkpoint,manifest)
        if not self.object_active:return
        from reproduction.scripts.training_checkpoint import cpu_tree
        path=Path(checkpoint).with_suffix('.object.pt');tmp=path.with_suffix('.tmp')
        torch.save(cpu_tree({'iteration':manifest['iteration'],'checkpoint_sha256':manifest['sha256'],
            'model':self.object_deform.object_model.state_dict(),'optimizer':self.object_optimizer.state_dict()}),tmp)
        tmp.replace(path)
        sidecar=Path(checkpoint).with_suffix('.metrics.json');saved=read_json(sidecar)
        saved['object_sha256']=sha256(path);write_json(sidecar,saved)

    def restore(self,checkpoint,iteration):
        super().restore(checkpoint,iteration)
        if not self.object_active:return
        path=Path(checkpoint).with_suffix('.object.pt')
        if not path.exists():
            receipt=read_json(self.run/'branch.json')
            if not receipt.get('initialize_object_at_branch') or iteration!=receipt['iteration'] or \
               iteration!=self.config.get('object_start',3000) or receipt['identical_full_checkpoint_sha256']!=sha256(checkpoint):
                raise ValueError('Missing object state outside receipted initial branch')
        else:
            if read_json(Path(checkpoint).with_suffix('.metrics.json')).get('object_sha256')!=sha256(path):
                raise ValueError('Object sidecar checksum mismatch')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['iteration']!=iteration or saved['checkpoint_sha256']!=sha256(checkpoint):raise ValueError('Wrong object checkpoint')
            self.object_deform.object_model.load_state_dict(saved['model'])
            self.object_optimizer.load_state_dict(saved['optimizer'])
            device=self.object_deform.scale_xyz.device
            for values in self.object_optimizer.state.values():
                for k,v in values.items():
                    if torch.is_tensor(v) and k!='step':values[k]=v.to(device)
        self.object_deform.object_iteration=iteration
