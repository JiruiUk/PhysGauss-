"""Post-hoc mechanism check: equal encoded time, distinct frozen trajectories.

Not an additional fitting run and not a global lower bound on the training set.
Queries only the frozen reconstruction, never images or physical GT.
"""
import copy
import json
from pathlib import Path
import runpy
import torch
from .run import ROOT,REPORT,initialize,read,write


def collision_pair(net):
    clock=copy.deepcopy(torch.nn.Sequential(net.time_embed,net.time_proj,torch.nn.Tanh())).cpu().double().eval()
    def value(t):
        with torch.no_grad():return clock(torch.as_tensor(t,dtype=torch.double).reshape(-1,1)).reshape(-1)
    ts=torch.linspace(0,1,10001,dtype=torch.double);q=value(ts)
    best=None
    for level in torch.linspace(q.min()+1e-7,q.max()-1e-7,101):
        edges=torch.where((q[:-1]-level)*(q[1:]-level)<0)[0]
        if len(edges)<2:continue
        def root(i):
            lo=ts[i].item();hi=ts[i+1].item();flo=value([lo]).item()-level.item()
            for _ in range(40):
                mid=(lo+hi)/2;fm=value([mid]).item()-level.item()
                if flo*fm<=0:hi=mid
                else:lo=mid;flo=fm
            return (lo+hi)/2
        first,last=root(edges[0]),root(edges[-1])
        if best is None or last-first>best[1]-best[0]:best=(first,last,level.item())
    return best


def main():
    np,torch,VelocityNet=initialize()
    from scene_PIDG.deform_model import DeformModel
    from variants.kinematic_tuning.model import center_trajectory,trajectory_velocity
    from .core import freeze,state_hash
    config=read(REPORT/'config.json');out=ROOT/config['output_run']
    result=read(out/'results.json') if (out/'results.json').exists() else {
        'arms':[{'arm':{'name':p.parent.name}} for p in sorted(out.glob('*/velocity_only.pt')) if not p.parent.name.startswith('_')]}
    data=torch.load(out/'targets.pt',map_location='cpu',weights_only=True)
    source=ROOT/config['source_run'];model=runpy.run_path(str(source/'model_config.py'))
    teacher=DeformModel(model['grid_args'],model['network_args'],reg_spatial_able=False,reg_temporal_able=False)
    teacher.load_weights(str(source),50000);freeze(teacher.grid4d);freeze(teacher.deform)
    hashes=[state_hash(m) for m in (teacher.grid4d,teacher.deform)]
    def trajectory(x,t):
        teacher.deform.attention_score=None
        return center_trajectory(x,teacher.step(x,t,fixed_attention=False)['d_xyz'])
    x=data['X'][:256].cuda();ids=data['particle_indices'][:256].cuda()
    checkpoints={'source':data['source_velocity']}
    for arm in result['arms']:
        name=arm['arm']['name']
        checkpoints[name]=torch.load(out/name/'velocity_only.pt',map_location='cpu',weights_only=False)['velocity_net']
    controls={}
    if (REPORT/'clock_control_result.json').exists():
        control=read(REPORT/'clock_control_result.json')
        if control['arm']['name'].startswith('C1_'):
            name=control['arm']['name'];ckpt=torch.load(out/name/'velocity_only.pt',map_location='cpu',weights_only=False)
            checkpoints[name]=ckpt['velocity_net'];controls[name]=ckpt['clock_mode']
    output={}
    for name,weights in checkpoints.items():
        net=VelocityNet().cuda();net.load_state_dict(weights);freeze(net)
        if name in controls:
            from .clock import apply_clock
            apply_clock(net,controls[name])
        pair=collision_pair(net)
        if pair is None:output[name]={'found_fold_collision':False};continue
        t1,t2,level=pair
        targets=[];preds=[];latent=[]
        with torch.no_grad():
            for tv in (t1,t2):
                t=x.new_full((len(x),1),tv)
                targets.append(trajectory_velocity(trajectory,x,t,config['derivative_h']))
                preds.append(net(torch.cat([x,t],-1),ids)[0])
                latent.append(torch.tanh(net.time_proj(net.time_embed(t[:1]))).item())
        p1,p2=preds;d1,d2=targets
        denom=d1.square().sum()+d2.square().sum()
        # Under an exactly shared output, the best vector is (d1+d2)/2.
        lower=(.5*(d1-d2).square().sum()/denom.clamp_min(1e-12)).sqrt().item()
        output[name]=dict(found_fold_collision=True,t1=t1,t2=t2,encoded_time_1=latent[0],encoded_time_2=latent[1],
            encoded_time_absolute_difference=abs(latent[0]-latent[1]),
            predicted_vector_difference_rms=(p1-p2).square().mean().sqrt().item(),
            target_vector_difference_rms=(d1-d2).square().mean().sqrt().item(),
            target_rms_1=d1.square().mean().sqrt().item(),target_rms_2=d2.square().mean().sqrt().item(),
            exact_shared_output_pair_relative_rmse_lower_bound=lower)
        print(name,output[name],flush=True)
    assert hashes==[state_hash(m) for m in (teacher.grid4d,teacher.deform)]
    write(REPORT/'time_collision_checks.json',dict(post_hoc=True,points=256,results=output,
          limitation='Paired continuous-time queries, not a global bound on fitted training samples; not physical velocity GT.',
          teacher_unchanged=True))


if __name__=='__main__':main()
