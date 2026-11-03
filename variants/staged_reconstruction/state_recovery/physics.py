"""Same-bank current-world neighborhoods; reject deficient spatial derivatives."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from ..stage3.physics import PhysicsField,material_acceleration,qualified_huber
from variants.coherent_momentum.operator import kinematics,momentum,interval
from variants.flow_comparison.temporal import position


def world_neighborhood(x,iteration,count=16,k=64):
    n=len(x);anchors=(torch.arange(min(count,n),device=x.device)*n//min(count,n)+iteration//100)%n
    tree=cKDTree(x.detach().cpu().numpy());_,ids=tree.query(x.detach().cpu().numpy()[anchors.cpu()],k=min(k+1,n),workers=1)
    # Remove the actual row ID, rather than assuming it sorts first on ties.
    ids=np.stack([row[row!=int(a)][:min(k,n-1)] for a,row in zip(anchors.cpu(),ids)])
    neighbors=torch.tensor(ids,device=x.device,dtype=torch.long)
    first=torch.unique(torch.cat([anchors,neighbors.flatten()]))
    return first,anchors,neighbors


class CurrentWorldField(PhysicsField):
    def loss(self,scene,t,iteration,samples=16,h=1/199):
        losses=[];diagnostics=[];tc=interval(t,h)[1]
        for k,obj in enumerate(scene.objects):
            with torch.no_grad():world=position(scene.deforms[k],obj.get_xyz,tc)+scene.centers[k]
            attempted=[]
            for support in [64,128,256]:
                first,anchors,neighbor=world_neighborhood(world,iteration,samples,support)
                query=obj.get_xyz[first];local=torch.searchsorted(first,anchors);ring=torch.searchsorted(first,neighbor)
                x,vtraj,atraj,_,_,times=kinematics(lambda s:position(scene.deforms[k],query,s)+scene.centers[k],t,h)
                velocity,stress=self.evaluate(k,query,times[1])
                acceleration,center=material_acceleration(lambda s:self.evaluate(k,query,s)[0],times[1],h)
                assert abs(center-times[1])<1e-10
                residual,valid,parts=momentum(x,velocity.double(),stress.double(),acceleration,self.rate_coefficient[k],local,ring)
                attempted.append({'neighbors':support,'valid_fraction':float(valid.float().mean())})
                if float(valid.float().mean())>=.25:break
            cmr,fraction=qualified_huber(residual,valid)
            consistency=torch.nn.functional.huber_loss(velocity.double(),vtraj)
            losses.append((consistency,cmr));rms=lambda a:float(a.detach().square().mean().sqrt())
            diagnostics.append({'object':k,'time_interval':times,'nodes':len(first),'anchors':len(anchors),
                'current_world_same_bank_neighbors':support,'support_attempts':attempted,'valid_fraction':fraction,
                'velocity_trajectory_RMS':rms(velocity-vtraj),'CMR_RMS':rms(residual[valid]),
                'trajectory_acceleration_RMS':rms(atraj[local]),'independent_acceleration_RMS':rms(acceleration[local]),
                'stress_pred_RMS':rms(stress),'stress_divergence_RMS':rms(parts['predicted'][valid]),
                'rate_divergence_RMS':rms(parts['rate'][valid]),
                'spatial_support_radius_mean':float((x[ring]-x[local,None]).norm(dim=-1).max(-1).values.mean()),
                'time_units':'sequence normalized; acquisition seconds unknown','not_independent_GT':True})
        return {name:torch.stack([v[i] for v in losses]).mean() for i,name in enumerate(['kinematic','CMR'])},diagnostics
