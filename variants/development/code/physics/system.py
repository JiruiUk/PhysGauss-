"""Small, switchable mechanics batches alongside the unchanged RGB reconstruction."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import torch
from torch import nn
from .config import load_config
from .differential import coordinates, derivative, jacobian
from .kinematics import official_position, compute_velocity_from_deformation
from .conservation import density_from_jacobian, mass_conservation_loss, volume_preservation_loss
from .constitutive import LinearElasticMaterial,NeoHookeanMaterial
from .constitutive.moe import ConstitutiveMoE
from .neural_residual import BoundedStressResidual
from .momentum import strong_form_momentum_residual
from .weak_form import weak_momentum_residual
from .contact import contact_force,plane_sdf
from .forces import external_force
from .rollout import rollout,closed_particle_force
from models.material_field import MaterialField,material_smoothness
from losses.uncertainty_flow import FlowUncertainty


def symmetric_stress(raw):
    assert raw.ndim==2 and raw.shape[1]==6
    a,b,c,d,e,f=raw.unbind(-1)
    return torch.stack([a,d,e,d,b,f,e,f,c],-1).reshape(-1,3,3)


class MechanicsSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=load_config()
        c=self.config
        self.register_buffer("velocity_closed",torch.tensor(False))
        self.register_buffer("kinematic_median",torch.tensor(0.))
        self.register_buffer("kinematic_passes",torch.tensor(0,dtype=torch.int64))
        if c.use_constitutive_model:
            self.linear=LinearElasticMaterial()
            self.neo=NeoHookeanMaterial()
        if c.use_material_field: self.material=MaterialField()
        if c.use_moe: self.moe=ConstitutiveMoE()
        if c.use_neural_residual: self.residual=BoundedStressResidual()
        if c.use_uncertainty_flow: self.flow_uncertainty=FlowUncertainty()

    def ramp(self,iteration,start):
        if os.environ.get("PIDG_UPGRADE_SMOKE")=="1": return 1.
        return max(0.,min(1.,(iteration-start)/2000))

    def stress(self,F,F_dot,X,features,iteration):
        c=self.config
        params=self.material(X,features) if c.use_material_field else None
        E,nu=(None,None) if params is None else (params["E"],params["nu"])
        if c.use_moe:
            stress,weights,_=self.moe(F,F_dot,X if params is None else params["features"],E,nu)
        else:
            transition=60 if os.environ.get("PIDG_UPGRADE_SMOKE")=="1" else c.constitutive_transition
            use_neo=c.constitutive_model=="neo_hookean" or (c.constitutive_model=="linear_to_neo" and iteration>=transition)
            self.active_constitutive = "neo" if use_neo else "linear"
            stress=(self.neo if use_neo else self.linear)(F,E,nu)
            weights=None
        penalty=F.sum()*0
        ratio=None
        if c.use_neural_residual:
            stress,penalty,ratio=self.residual(stress,F,X)
        assert stress.shape==(len(X),3,3)
        return stress,params,weights,penalty,ratio

    def forward(self,gaussians,deform,time,iteration,model_path):
        c=self.config
        if not c.use_kinematic_closure: return gaussians.get_xyz.sum()*0,{}
        with torch.no_grad():
            indices=torch.randperm(len(gaussians.get_xyz),device=gaussians.get_xyz.device)[:c.physics_samples]
        X,t=coordinates(gaussians.get_xyz[indices].detach(),time)
        features=gaussians.get_features[indices,0,:].detach()
        phi=lambda X,t:official_position(deform,X,t,True)
        def evaluate(X,t):
            x=phi(X,t)
            v=torch.cat([derivative(x[:,i],t) for i in range(3)],-1)
            a=torch.cat([derivative(v[:,i],t) for i in range(3)],-1) if c.use_acceleration else None
            F=jacobian(x,X) if c.use_deformation_gradient else None
            J=torch.linalg.det(F)[:,None] if F is not None else None
            return x,v,a,F,J
        x,v,a,F,J=evaluate(X,t)
        network_velocity,raw_stress=gaussians.velocity_net(torch.cat([X,t],-1),gaussians.particle_indices[indices])
        kin=(network_velocity-v).square().mean()
        relative=(network_velocity-v).norm(dim=-1)/(v.norm(dim=-1)+1e-6)
        self.kinematic_median.copy_(relative.detach().median())
        if self.kinematic_median<.1:
            self.kinematic_passes.add_(1)
        else:
            self.kinematic_passes.zero_()
        if self.kinematic_passes>=3: self.velocity_closed.fill_(True)
        loss=c.lambda_kin*self.ramp(iteration,3000)*kin
        diagnostics={key:None for key in ("loss_rgb","loss_flow","loss_kin","loss_momentum","loss_mass",
                     "loss_volume","mean_speed","mean_acceleration","mean_detF","min_detF","max_detF","mean_stress_norm")}
        diagnostics.update(loss_kin=kin,kinematic_relative_median=relative.median(),mean_speed=v.norm(dim=-1).mean())
        if a is not None: diagnostics["mean_acceleration"]=a.norm(dim=-1).mean()
        if F is not None:
            invalid=J.detach()<=0
            diagnostics.update(mean_detF=J.mean(),min_detF=J.min(),max_detF=J.max(),
                               nonpositive_detF_count=invalid.sum(),quadrature_count=len(J))
            barrier=torch.relu(.05-J).square().mean()
            loss=loss+c.lambda_jacobian*self.ramp(iteration,5000)*barrier
            if invalid.any() and (c.use_constitutive_model or c.use_mass_conservation):
                # Report originals and re-evaluate valid reference particles; no silent density clamp.
                valid=~invalid[:,0]
                diagnostics["excluded_invalid_particles"]=invalid.sum()
                if not valid.any(): return loss,diagnostics
                indices=indices[valid]
                features=features[valid]
                X,t=coordinates(X.detach()[valid],t.detach()[valid])
                x,v,a,F,J=evaluate(X,t)
                _,raw_stress=gaussians.velocity_net(torch.cat([X,t],-1),gaussians.particle_indices[indices])
            if c.use_mass_conservation:
                if (J.detach()<=0).any():
                    diagnostics["mass_unavailable_invalid_J"]=True
                    return loss,diagnostics
                rho=density_from_jacobian(c.rho0,J)
                mass=mass_conservation_loss(rho,J,c.rho0)
                diagnostics["loss_mass"]=mass
                loss=loss+c.lambda_mass*self.ramp(iteration,5000)*mass
            else: rho=X.new_full((len(X),1),c.rho0)
            if c.use_volume_preservation:
                volume=volume_preservation_loss(J)
                diagnostics["loss_volume"]=volume
                loss=loss+c.lambda_volume*self.ramp(iteration,5000)*volume
        if c.use_constitutive_model:
            F_dot=torch.stack([torch.cat([derivative(F[:,i,j],t) for j in range(3)],-1)
                               for i in range(3)],1) if c.use_moe else torch.zeros_like(F)
            stress,params,weights,penalty,ratio=self.stress(F,F_dot,X,features,iteration)
            constitutive=(symmetric_stress(raw_stress)-stress).square().mean()
            diagnostics.update(mean_stress_norm=stress.norm(dim=(-2,-1)).mean(),loss_constitutive=constitutive)
            loss=loss+c.lambda_constitutive*self.ramp(iteration,5000)*constitutive
            if params is not None:
                smooth=material_smoothness(X,params["features"])
                prior=params["E"].log().square().mean()
                loss=loss+self.ramp(iteration,5000)*(c.lambda_material_smooth*smooth+1e-4*prior)
                diagnostics.update(mean_E=params["E"].mean(),mean_nu=params["nu"].mean(),loss_material_smooth=smooth)
            else:
                E,nu=(self.neo if getattr(self,"active_constitutive","linear")=="neo" else self.linear).parameters_physical()
                diagnostics.update(mean_E=E,mean_nu=nu)
            if weights is not None:
                entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1).mean()
                diagnostics["mixture_entropy"]=entropy
                loss=loss+1e-4*self.ramp(iteration,20000)*entropy
            if ratio is not None:
                diagnostics["neural_stress_ratio_median"]=ratio.median()
                loss=loss+c.lambda_residual*self.ramp(iteration,20000)*penalty
            contact=None
            if c.use_contact:
                contact,collision=contact_force(x,plane_sdf(x,offset=c.contact_offset),c.contact_stiffness)
                loss=loss+c.lambda_contact*self.ramp(iteration,20000)*collision
                diagnostics["loss_contact"]=collision
            force=external_force(x,rho,c.gravity_vector if c.gravity_enabled else None,contact)
            diagnostics.update(mean_density=rho.mean(),mean_body_force=force.norm(dim=-1).mean())
            if c.momentum_mode=="closed_strong":
                residual=strong_form_momentum_residual(rho,a,stress,X,force,F)
                momentum=residual.square().mean()
                loss=loss+c.lambda_momentum*self.ramp(iteration,10000)*momentum
                diagnostics["loss_momentum"]=momentum
            elif c.momentum_mode=="weak":
                volumes=J/len(X)
                residual=weak_momentum_residual(x,rho,a,stress,volumes,force,c.weak_form_k,c.weak_form_bandwidth_scale)
                momentum=residual.square().mean()
                loss=loss+c.lambda_weak_momentum*self.ramp(iteration,15000)*momentum
                diagnostics.update(loss_momentum=momentum,loss_weak_momentum=momentum)
            if c.use_rollout and float(t.detach()[0])<.995:
                dt=min(.001,(1-float(t.detach()[0]))/4)
                vol=X.new_full((len(X),1),1/len(X))
                def material(f): return self.stress(f,torch.zeros_like(f),X,features,iteration)[0]
                def force_model(position,velocity,tau):
                    return closed_particle_force(X,position,vol,material,
                        c.gravity_vector if c.gravity_enabled else None,
                        contact={"plane":{"offset":c.contact_offset},"stiffness":c.contact_stiffness} if c.use_contact else None, rho0=c.rho0)
                positions,_=rollout(x,v,c.rho0*vol,force_model,dt,4)
                target=phi(X,t+4*dt)
                forecasting=(positions[-1]-target).square().mean()
                loss=loss+c.lambda_rollout*self.ramp(iteration,25000)*forecasting
                diagnostics["loss_rollout"]=forecasting
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite closed-mechanics loss")
        return loss,diagnostics

    def flow_velocity(self,gaussians,deform,time):
        source=self.config.velocity_source
        if source=="network" or (source=="gated_deformation" and not self.velocity_closed):
            return gaussians.get_velocity_and_stress(time)[0]
        velocity=[]
        for X in gaussians.get_xyz.split(4096):
            velocity.append(compute_velocity_from_deformation(
                lambda X,t:official_position(deform,X,t,False),X.detach(),time))
        return torch.cat(velocity)

    def write_diagnostics(self,model_path,iteration,diagnostics,rgb,flow=None):
        record={"iteration":iteration,**diagnostics,"loss_rgb":float(rgb.detach()),
                "loss_flow":None if flow is None else float(flow.detach()),
                "velocity_source":self.config.velocity_source,"velocity_gate_passed":bool(self.velocity_closed)}
        for key,value in list(record.items()):
            if torch.is_tensor(value): record[key]=value.detach().item()
        record["flow_valid_fraction"]=getattr(self,"last_flow_valid_fraction",None)
        record["flow_epe_pixels"]=getattr(self,"last_flow_epe",None)
        for key,value in list(record.items()):
            if torch.is_tensor(value): record[key]=value.detach().item()
        path=Path(model_path)/"physics.jsonl"
        with path.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
