"""Original PIDG fields with tested current-world material momentum discretization.

No constitutive law, body force, contact, or material parameter is introduced.
Time is sequence-normalized; no acquisition seconds or physical units are known.
"""
import torch
from torch import nn
from variants.flow_comparison.local_momentum import Graph
from variants.coherent_momentum.operator import interval,kinematics,momentum
from variants.flow_comparison.temporal import position


def material_acceleration(velocity,t,h):
    """D v(X,t)/Dt at fixed material coordinate. Do not add convection again."""
    tm,tc,tp=interval(t,h)
    return (velocity(tp).double()-velocity(tm).double())/(tp-tm),tc


def qualified_huber(residual,valid,minimum_fraction=.25):
    fraction=float(valid.float().mean())
    if fraction<minimum_fraction or not bool(valid.any()):
        raise ValueError(f'Insufficient nondegenerate same-object neighborhoods: {fraction:.4f}')
    return torch.nn.functional.huber_loss(residual[valid],torch.zeros_like(residual[valid])),fraction


class PhysicsField(nn.Module):
    def __init__(self,object_count):
        super().__init__()
        from motion_utils.pidg import VelocityNet
        # Original class, shared per-object planes/heads; no per-particle free field.
        self.fields=nn.ModuleList([VelocityNet(enable_index_embedding=False).cuda() for _ in range(object_count)])
        self.register_buffer('rate_coefficient',torch.ones(object_count,device='cuda'))
        self.graphs=[Graph() for _ in range(object_count)]

    def evaluate(self,k,X,t):
        return self.fields[k](torch.cat([X,X.new_full((len(X),1),float(t))],-1))

    def loss(self,scene,t,iteration,samples=16,h=1/199):
        losses=[];diagnostics=[]
        for k,obj in enumerate(scene.objects):
            # Build/query each graph separately. No static or other-object point
            # can enter a derivative neighborhood, including unknown image IDs.
            _,first,anchors,_,neighbor=self.graphs[k].sample(obj.get_xyz,iteration,count=samples,k=32)
            query=obj.get_xyz[first]
            local=torch.searchsorted(first,anchors);ring=torch.searchsorted(first,neighbor)
            x,vtraj,atraj,_,_,times=kinematics(lambda s:position(scene.deforms[k],query,s)+scene.centers[k],t,h)
            velocity,stress=self.evaluate(k,query,times[1])
            acceleration,center=material_acceleration(lambda s:self.evaluate(k,query,s)[0],times[1],h)
            assert abs(center-times[1])<1e-10
            residual,valid,parts=momentum(x,velocity.double(),stress.double(),acceleration,
                self.rate_coefficient[k],local,ring)
            cmr,fraction=qualified_huber(residual,valid)
            consistency=torch.nn.functional.huber_loss(velocity.double(),vtraj)
            losses.append((consistency,cmr))
            rms=lambda a:float(a.detach().square().mean().sqrt())
            diagnostics.append({'object':k,'time_interval':times,'nodes':len(first),'anchors':len(anchors),
                'valid_fraction':fraction,'velocity_trajectory_RMS':rms(velocity-vtraj),
                'CMR_RMS':rms(residual[valid]),'trajectory_acceleration_RMS':rms(atraj[local]),
                'independent_acceleration_RMS':rms(acceleration[local]),'stress_pred_RMS':rms(stress),
                'stress_divergence_RMS':rms(parts['predicted'][valid]),'rate_divergence_RMS':rms(parts['rate'][valid]),
                'time_units':'sequence normalized; acquisition seconds unknown','not_independent_GT':True})
        return {name:torch.stack([l[i] for l in losses]).mean() for i,name in enumerate(['kinematic','CMR'])},diagnostics
