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
        if self.config.get('interface_enabled') and not (self.config.get('constitutive') and self.config.get('multi_material')):
            raise ValueError('Interface traction requires constitutive consistency and known materials')
        self.run=Path(run)
        self.train_count=train_count
        self.optimizer=None
        self.material_ids=None
        self.material_ids_hash=None
        self.force_enabled=self.config.get('structured_force',False)
        self.force=None;self.force_optimizer=None
        if self.force_enabled:
            if not self.enabled:raise ValueError('Force integration requires F diagnostics in the supported ladder')
            from .forces import StructuredForce
            self.force=StructuredForce(self.config)
            if list(self.force.parameters()):
                self.force_optimizer=torch.optim.Adam(self.force.parameters(),lr=self.config.get('force_lr',1e-4))
        self.force_iteration=None
        self.last_force_regularizer=None
        self.neural_step_enabled=True;self.densification_enabled=True;self.force_active=self.force_enabled
        self.schedule=None
        if self.config.get('staged_optimization'):
            from .staging import StagedOptimization
            self.schedule=StagedOptimization(self.config,self.run)
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

    def before_iteration(self,iteration,gaussians,deform,opt):
        if self.schedule is not None:self.schedule.apply(iteration,gaussians,deform,self)

    def loss(self,iteration,gaussians,deform,d_xyz,fid,opt,cauchy_loss=None):
        if not self.enabled or iteration % self.config.get('interval',100):
            return None
        if self.schedule is not None and self.schedule.last_stage=='A':return None
        if self.config.get('constitutive') and opt is not None and iteration<=getattr(opt,'warm_up',0):return None
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
            if self.schedule is not None:record['optimization_stage']=self.schedule.last_stage
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
                if self.config.get('interface_enabled'):
                    from .interface import interface_pairs,traction_loss
                    reverse=torch.full((len(X),),-1,device=X.device,dtype=torch.long)
                    reverse[ids]=torch.arange(len(ids),device=X.device)
                    pairs=interface_pairs(X[ids],labels.to(X.device),reverse[graph[ids]],self.config.get('interface_max_distance',float('inf')))
                    interface=traction_loss(pred,X[ids],pairs,result['confidence'])
                    interface_weighted=interface*self.config.get('interface_weight',1e-4)
                    weighted=weighted+interface_weighted
                    record.update(interface_pairs=len(pairs),loss_interface_raw=interface.item() if len(pairs) else None,
                                  loss_interface_weighted=interface_weighted.item(),interface_normal_convention='shared canonical centre-line normal')
        if self.force_enabled and self.force_iteration==iteration:
            with torch.enable_grad():
                weighted=self.last_force_regularizer if weighted is None else weighted+self.last_force_regularizer
            record.update(self.last_force_stats,loss_force_regularization=self.last_force_regularizer.item(),force_time_convention='fid; original velocity/CMR query remains fid/train_count')
        with (self.run/'extension.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        return weighted

    def optimizer_step(self):
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        if self.force_optimizer is not None:
            self.force_optimizer.step();self.force_optimizer.zero_grad(set_to_none=True)

    def momentum_loss(self,gaussians,coords,dt,rho,sample_size=20000,chunk_size=5000):
        from .forces import momentum_components
        N=len(coords);M=min(N,sample_size);perm=torch.randperm(N,device=coords.device)[:M]
        total=coords.new_zeros(());magnitude=coords.new_zeros(());sparsity=coords.new_zeros(())
        stats={key:0. for key in ('inertial_term_norm','stress_divergence_norm','external_force_norm','total_residual_norm')}
        for start in range(0,M,chunk_size):
            selected=perm[start:start+chunk_size];chunk=coords[selected];ids=gaussians.particle_indices[selected]
            inertial,divergence=momentum_components(chunk,lambda value:gaussians.velocity_net(value,ids),gaussians.force_sigma,rho)
            force=self.force(chunk[:,:3],self.force_time,ids);residual=inertial-divergence-force
            weight=len(selected)/M;total=total+residual.square().mean()*weight
            magnitude=magnitude+force.square().sum(-1).mean()*weight
            sparsity=sparsity+force.abs().sum(-1).mean()*weight
            for name,value in zip(stats,(inertial,divergence,force,residual)):
                stats[name]+=value.detach().norm(dim=-1).mean().item()*weight
            torch.cuda.empty_cache()
        temporal=self.force.coefficients.diff(dim=0).square().mean().to(coords) if self.force.coefficients is not None else coords.new_zeros(())
        weights=[self.config.get('force_magnitude_weight',1e-4),self.config.get('force_temporal_weight',1e-3),self.config.get('force_sparsity_weight',0.)]
        if min(weights)<0:raise ValueError('Force regularization weights must be nonnegative')
        self.last_force_regularizer=weights[0]*magnitude+weights[1]*temporal+weights[2]*sparsity
        self.last_force_stats={**stats,'force_parameter_count':sum(p.numel() for p in self.force.parameters()),
                               'force_magnitude_raw':magnitude.item(),'force_temporal_raw':temporal.item(),'force_sparsity_raw':sparsity.item(),
                               'cauchy_sampling_factor':M/N}
        return total*(M/N)

    def checkpoint_state(self):
        if not self.enabled:return None
        from reproduction.scripts.training_checkpoint import cpu_tree
        return cpu_tree({'config':self.config,'train_count':self.train_count,
                         'material':self.material.state_dict() if self.material is not None else None,
                         'optimizer':self.optimizer.state_dict() if self.optimizer is not None else None,
                         'material_ids':self.material_ids,'material_ids_hash':self.material_ids_hash,
                         'force':self.force.state_dict() if self.force is not None else None,
                         'force_optimizer':self.force_optimizer.state_dict() if self.force_optimizer is not None else None,
                         'force_inputs':self.force.input_hashes if self.force is not None else {},
                         'schedule':self.schedule.state() if self.schedule is not None else None,
                         'neighbors':self.graph.neighbors,'graph_iteration':self.graph.last_iteration})

    def restore_checkpoint(self,state,device=None):
        if state is None:
            if self.enabled:raise ValueError('Missing extension checkpoint state')
            return
        if state['config']!=self.config or state['train_count']!=self.train_count:
            raise ValueError('Extension configuration differs from checkpoint')
        if state.get('material_ids_hash')!=self.material_ids_hash:
            raise ValueError('Material annotation changed since checkpoint')
        if self.force is not None:
            if state.get('force_inputs')!=self.force.input_hashes:raise ValueError('Force input changed since checkpoint')
            self.force.load_state_dict(state['force'])
        if self.force_optimizer is not None:self.force_optimizer.load_state_dict(state['force_optimizer'])
        if self.schedule is not None:self.schedule.restore(state.get('schedule'))
        if self.material is not None:self.material.load_state_dict(state['material'])
        if self.optimizer is not None:self.optimizer.load_state_dict(state['optimizer'])
        self.graph.neighbors=state['neighbors'].to(device) if state['neighbors'] is not None else None
        self.graph.last_iteration=state['graph_iteration']
