"""Optional coherent momentum objective with the existing M1 gradient policy."""
import torch
import torch.nn.functional as F
from variants.momentum_guard.runtime import Tuning as Parent
from variants.flow_comparison.temporal import position
from variants.unified_flow.model import predict_stress
from variants.momentum_guard.balance import balance
from .operator import kinematics, momentum


class Tuning(Parent):
    def __init__(self,*args):
        super().__init__(*args)
        self.coherent = self.enabled and self.config.get('coherent_momentum',False)
        if self.coherent and (not self.guard or self.stress_source!='shared'):
            raise ValueError('Coherent experiment requires the M1 guard/shared architecture')

    def momentum_loss(self,iteration,gaussians,deform,fid):
        start=self.config.get('momentum_start',8000)
        if not self.coherent or iteration<=start:
            return super().momentum_loss(iteration,gaussians,deform,fid)
        if self._reference is None or self._reference[0]!=iteration:
            raise RuntimeError('Coherent momentum needs same-step RGB reference')
        _,image,weight=self._reference; self._reference=None
        canonical=gaussians.get_xyz.detach()
        _,first,anchors,_,neighbor=self.graph.sample(canonical,iteration,
            count=self.config.get('momentum_samples',64),k=self.config.get('coherent_neighbors',32))
        query=canonical[first]
        local=torch.searchsorted(first,anchors); ring=torch.searchsorted(first,neighbor)
        t=float(fid.item()); h=float(self.config.get('coherent_dt',1/199))
        now,v,a,xm,xp,times=kinematics(lambda s:position(deform,query,s),t,h)
        stress=predict_stress(deform,query,times[1]).double()
        r,valid,parts=momentum(now,v,stress,a,gaussians.force_sigma,local,ring,
            self.config.get('coherent_ridge',1e-5))
        if float(valid.float().mean())<.25:
            raise ValueError('Insufficient full-rank quadratic neighborhoods')
        raw=F.huber_loss(r[valid],torch.zeros_like(r[valid]))
        if not bool(raw.isfinite()): raise FloatingPointError('Nonfinite coherent momentum')
        ramp=min(1.,(iteration-start)/max(1,self.config.get('momentum_ramp',4000)))
        # xp_physics=2*x-xm+h^2*div(sigma); same particle / interval / trajectory.
        # Train in acceleration units, avoiding a deceptively small h^2 loss.
        rollout=2*now[local]-xm[local]+h*h*(a[local]-r)
        rms=lambda z:float(z.detach().square().mean().sqrt())
        self.append('coherent_momentum.jsonl',{'iteration':iteration,'requested_t':t,'interval_times':times,
            'h':h,'raw_huber':float(raw.detach()),'residual_rms':rms(r[valid]),
            'rollout_position_rms':rms((xp[local]-rollout)[valid]),
            'velocity_impulse_rms':h*rms(r[valid]),'valid_fraction':float(valid.float().mean()),
            'predicted_divergence_rms':rms(parts['predicted'][valid]),
            'rate_divergence_rms':rms(parts['rate'][valid]),'ramp':ramp,'nodes':len(first),
            'min_valid_rank_ratio':float(parts['rank_ratio'][valid].min()),
            'boundary_center_shift':times[1]-t})
        self.audit('momentum',raw,gaussians,deform,iteration)
        value=raw*ramp
        if self.config.get('verify_capture_raw',False): self._verification_raw=value
        groups={'grid':list(deform.grid4d.parameters()),
            'deformation':[p for n,p in deform.deform.named_parameters() if not n.startswith('shared_stress.')]}
        private=list(deform.deform.shared_stress.parameters())+[gaussians.force_sigma]
        guarded,info=balance(image,value,groups,private,
            float(self.config.get('momentum_shared_ratio',.1)),weight)
        self.append('momentum_gradient_guard.jsonl',{'iteration':iteration,
            'true_momentum_forward':float(value.detach()),'image_forward':float(image.detach()),
            'physics_weight':weight,**info})
        return guarded
