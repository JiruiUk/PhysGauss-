"""Training hooks; disabled mode creates no parameters and consumes no RNG."""
import json
from pathlib import Path
import torch
from .deformation_gradient import CanonicalGraph, estimate_F


class Extension:
    def __init__(self,config_path,run,train_count=1):
        self.config=json.loads(Path(config_path).read_text()) if config_path else {}
        self.enabled=self.config.get('deformation_gradient',False)
        if self.config.get('constitutive') and not self.enabled:
            raise ValueError('Constitutive consistency requires the F estimator')
        self.run=Path(run)
        self.train_count=train_count
        self.optimizer=None
        self.material_ids=None
        self.material_ids_hash=None
        if self.enabled:
            self.graph=CanonicalGraph(self.config.get('num_neighbors',16))
        self.material=None
        if self.config.get('constitutive_diagnostics') or self.config.get('constitutive'):
            from .constitutive import MaterialParameters,NeoHookean
            if self.config.get('multi_material'):
                import hashlib
                from .materials import MaterialBank,load_material_ids
                path=Path(self.config['material_ids_file'])
                if not path.is_absolute():path=Path(__file__).resolve().parents[1]/path
                self.material=MaterialBank(self.config['regions'],self.config.get('learn_material_parameters',False))
                self.material_ids=load_material_ids(path,len(self.config['regions']))
                self.material_ids_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                self.material=MaterialParameters(self.config.get('mu_init',1.),self.config.get('kappa_init',10.),
                                                 self.config.get('learn_material_parameters',False))
            self.constitutive=NeoHookean()
            parameters=[p for p in self.material.parameters() if p.requires_grad]
            if parameters:
                self.optimizer=torch.optim.Adam(parameters,lr=self.config.get('material_lr',1e-5))

    def loss(self,iteration,gaussians,deform,d_xyz,fid,opt,cauchy_loss=None):
        if not self.enabled or iteration % self.config.get('interval',100):
            return None
        from utils.general_utils import quaternion_to_matrix
        active=self.config.get('constitutive',False)
        weighted=None
        with torch.enable_grad() if active else torch.no_grad():
            X=gaussians.get_xyz
            x=X if not torch.is_tensor(d_xyz) else (quaternion_to_matrix(d_xyz[...,:4])@X[...,None]).squeeze(-1)+d_xyz[...,4:]
            graph=self.graph.get(X,iteration)
            ids=torch.arange(0,len(X),max(1,len(X)//256),device=X.device)[:256]
            result=estimate_F(X,x,graph,ids,min_determinant=self.config.get('min_determinant',1e-6))
            record={'iteration':iteration,'mean_det_F':result['J'].mean().item(),
                    'min_det_F':result['J'].min().item(),'invalid_F_fraction':1-result['valid'].float().mean().item(),
                    'diagnostic_queries':len(ids),'new_loss_added':False}
            if self.material is not None:
                valid=result['valid']
                if self.material_ids is None:
                    params=self.material()
                    record.update(mu=params['mu'].item(),kappa=params['kappa'].item())
                else:
                    particle_ids=gaussians.particle_indices[ids].to(self.material_ids.device)
                    if particle_ids.min()<0 or particle_ids.max()>=len(self.material_ids):
                        raise ValueError('No known material label for a particle ID')
                    labels=self.material_ids[particle_ids]
                    params=self.material(labels)
                    record.update(mu=None,kappa=None,material_ids_present=labels.unique().tolist(),
                                  material_parameters={k:v.detach().tolist() for k,v in self.material.values().items()})
                if not active:
                    selected={k:v[valid.to(v.device)] if v.ndim else v for k,v in params.items()}
                    sigma=self.constitutive(result['F'][valid],selected)
                    record['mean_sigma_const_norm']=sigma.norm(dim=(-1,-2)).mean().item() if valid.any() else None
            if active:
                from .losses import predicted_stress,constitutive_consistency
                coords=torch.cat((X[ids],fid.expand(len(ids),1)),dim=-1).requires_grad_(True)
                pred=predicted_stress(coords,gaussians.particle_indices[ids],gaussians.velocity_net,gaussians.force_sigma)
                raw,const=constitutive_consistency(pred,result['F'],params,result['confidence'],valid)
                weight=self.config.get('loss_weight',1e-4)
                if weight<0:raise ValueError('Negative constitutive weight')
                weighted=raw*weight
                record.update(new_loss_added=weight>0,loss_constitutive_raw=raw.item(),loss_constitutive_weighted=weighted.item(),
                              mean_sigma_const_norm=const[valid].norm(dim=(-1,-2)).mean().item() if valid.any() else None,
                              mean_sigma_pred_norm=pred.norm(dim=(-1,-2)).mean().item(),
                              mean_sigma_difference_norm=(pred[valid]-const[valid]).norm(dim=(-1,-2)).mean().item() if valid.any() else None,
                              cauchy_residual_legacy=cauchy_loss.detach().item() if cauchy_loss is not None else None,
                              stress_time_convention='fid, matching rendered F',force_sigma=gaussians.force_sigma.item())
                gradients=torch.autograd.grad(weighted,(result['F'],pred),retain_graph=True,allow_unused=True)
                record['constitutive_F_gradient_norm']=gradients[0].norm().item() if gradients[0] is not None else 0.
                record['constitutive_predicted_stress_gradient_norm']=gradients[1].norm().item() if gradients[1] is not None else 0.
        with (self.run/'extension.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        return weighted

    def optimizer_step(self):
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def checkpoint_state(self):
        if not self.enabled:return None
        from reproduction.scripts.training_checkpoint import cpu_tree
        return cpu_tree({'config':self.config,'train_count':self.train_count,
                         'material':self.material.state_dict() if self.material is not None else None,
                         'optimizer':self.optimizer.state_dict() if self.optimizer is not None else None,
                         'material_ids':self.material_ids,'material_ids_hash':self.material_ids_hash,
                         'neighbors':self.graph.neighbors,'graph_iteration':self.graph.last_iteration})

    def restore_checkpoint(self,state,device=None):
        if state is None:
            if self.enabled:raise ValueError('Missing extension checkpoint state')
            return
        if state['config']!=self.config or state['train_count']!=self.train_count:
            raise ValueError('Extension configuration differs from checkpoint')
        if state.get('material_ids_hash')!=self.material_ids_hash:
            raise ValueError('Material annotation changed since checkpoint')
        if self.material is not None:self.material.load_state_dict(state['material'])
        if self.optimizer is not None:self.optimizer.load_state_dict(state['optimizer'])
        self.graph.neighbors=state['neighbors'].to(device) if state['neighbors'] is not None else None
        self.graph.last_iteration=state['graph_iteration']
