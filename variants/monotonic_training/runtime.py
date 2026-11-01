"""Reuse H3 and existing joint kinematic loss; add clock-contract diagnostics."""
import torch
from variants.integrity_tuning.runtime import Tuning as Parent
from variants.metric_tuning.runtime import write_json


class Tuning(Parent):
    def after_scene(self,gaussians,deform):
        super().after_scene(gaussians,deform)
        if not self.enabled:return
        mode=self.config.get('velocity_clock_mode','legacy')
        if getattr(gaussians.velocity_net,'velocity_clock_mode','legacy')!=mode:raise ValueError('Clock mode was not configured')
        write_json(self.run/'clock_contract.json',{'velocity_clock_mode':mode,
          'mapping':'tanh(2*t-1)' if mode=='fixed_monotonic' else 'original Fourier/projected clock',
          'coordinate':'canonical X','time':'unchanged common normalized frame time',
          'parameter_keys_and_optimizer_groups_preserved':True,'restore_requires_this_run_config':True,
          'joint_kinematic_weight':self.config.get('kinematic_weight',0.)})

    def after_update(self,iteration,gaussians):
        super().after_update(iteration,gaussians)
        if not self.enabled or self.config.get('velocity_clock_mode','legacy')!='fixed_monotonic':return
        if iteration%1000==0 or self.config.get('engineering_smoke',False):
            net=gaussians.velocity_net
            frozen=[p for module in (net.time_embed,net.time_proj) for p in module.parameters()]
            if any(p.requires_grad or p.grad is not None for p in frozen):raise ValueError('Unused clock received gradient')
            self.append('clock_audit.jsonl',{'iteration':iteration,'clock':'fixed_monotonic',
               'unused_clock_parameters_frozen':True,'velocity_head_finite':bool(torch.isfinite(net.velocity_head.weight).all())})
