"""Native disabled parity, shared mapping and full controlled-run audit."""
import argparse
import json
import random
import numpy as np
import torch
from ..prepare_data import ROOT,REPORT,DATA,read,write,sha
from ..train_static import camera
from ..stage2.train import image
from ..union.train import reconstruction
from .prepare import EXP,WORK,INITIAL,INITIAL_SHA
from .train import load_initial,step_loss
from .runtime import install,renderer_for


def native():
    torch.set_num_threads(4)
    renderer,scene,opt,initial,base=load_initial()
    plan=read(EXP/'protocol.json'); meta=read(DATA/'transforms_train.json')
    for index,frame in enumerate(meta['frames']): frame['_index']=index
    frame=meta['frames'][0]; rgba=image(frame,800); cam=camera(frame,meta,800)
    gt=torch.tensor(rgba[...,:3]*rgba[...,3:4],device='cuda').permute(2,0,1)
    original_deform=scene.deforms[0]
    original=scene.render(renderer,cam,frame['time'])
    expected=reconstruction(original,gt,rgba,scene,cam)
    expected.backward(); gradients={name:None if p.grad is None else p.grad.clone() for name,p in scene.named_parameters()}
    assert install(scene,{},False) is None
    assert install(scene,dict(plan,covariance_transport_enabled=False),True) is None
    assert scene.deforms[0] is original_deform
    disabled=scene.render(renderer,cam,frame['time'])
    assert torch.equal(original['render'],disabled['render']) and torch.equal(original['alpha'],disabled['alpha'])
    loss,_=step_loss(scene,renderer,opt,frame,meta,{},1,plan,base,False)
    assert torch.equal(loss,expected); loss.backward(); error,scale=0.,0.
    for name,p in scene.named_parameters():
        if gradients[name] is None: assert p.grad is None
        else:
            error=max(error,float((p.grad-gradients[name]).abs().max())); scale=max(scale,float(gradients[name].abs().max()))
    assert error<max(1e-7,scale*2e-5)
    parameter_ids=[id(p) for p in scene.parameters()]
    cpu_rng=torch.get_rng_state().clone(); cuda_rng=[value.clone() for value in torch.cuda.get_rng_state_all()]
    adapter=install(scene,plan,True,opt)
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    assert set(parameter_ids).issubset({id(p) for p in scene.parameters()})
    assert sum(p.numel() for p in adapter.motion.parameters())>0
    active=scene.render(renderer,cam,frame['time'])
    enabled_error={key:{'max':float((active[key]-original[key]).abs().max()),'mean':float((active[key]-original[key]).abs().mean())} for key in ['render','alpha']}
    assert all(row['max']<plan['initial_native_max_abs_tolerance'] and row['mean']<plan['initial_native_mean_abs_tolerance'] for row in enabled_error.values()),enabled_error
    renderer=renderer_for(renderer)
    from ..train_static import PIPE
    dynamic=scene.dynamic_at(frame['time'])
    dispatched=renderer(cam,dynamic,PIPE,scene.centers.new_zeros(3),0.,0.,0.)
    assert dispatched['alpha'].isfinite().all() and dynamic.transported_covariance.shape==(len(dynamic.get_xyz),6)
    calls=[]; old_step=original_deform.step
    def traced(x,t,fixed_attention=False):
        calls.append(len(x)); return old_step(x,t,fixed_attention=fixed_attention)
    original_deform.step=traced
    try: scene.dynamic_at(.43)
    finally: original_deform.step=old_step
    assert calls==[],calls
    cache={}; rows=[]
    for local in range(1,21):
        loss,row=step_loss(scene,renderer,opt,meta['frames'][local-1],meta,cache,16000+local,plan,base,True)
        loss.backward(); assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
        assert all(p.grad is None for p in list(scene.grids.parameters())+list(scene.networks[0].parameters()))
        opt.step(); scene.constrain(); rows.append(row)
    assert sum('physics' in row for row in rows)==2
    from variants.flow_comparison.temporal import position
    mapping={}
    with torch.no_grad():
        for t in [0.,.43,1.]:
            direct=scene.dynamic_at(t).get_xyz
            queried=position(adapter,scene.pools[0].get_xyz,t)+scene.centers[0]
            value=float((direct-queried).abs().max()); assert value<3e-5
            mapping[str(t)]=value
    opt.zero_grad(set_to_none=True)
    x=scene.pools[0].get_xyz
    mapped=scene.dynamic_at(.43).get_xyz
    grad=torch.autograd.grad(mapped.square().mean(),[x,*scene.grids.parameters(),*scene.networks.parameters()],allow_unused=True)
    assert grad[0].isfinite().all() and grad[0].abs().sum()>0
    assert all(g is None or g.isfinite().all() for g in grad)
    assert sum(float(g.abs().sum()) for g in grad[1:] if g is not None)>0
    opt.zero_grad(set_to_none=True)
    from .core import motion_with_differential,covariance,transport
    from ..geometry_recovery.model import bounded_scale
    pool=scene.pools[0]; query=pool.get_xyz[:128]
    _,jac=motion_with_differential(adapter.motion,query,query.new_full((len(query),1),.43))
    reference=covariance(bounded_scale(pool._scaling[:128],torch.zeros_like(pool._scaling[:128]),pool.spacing,True),pool.get_rotation[:128])
    cov_loss=transport(jac,reference).square().mean()
    groups={'canonical_xyz':[pool._xyz],'canonical_scale':[pool._scaling],
            'canonical_rotation':[pool._rotation],'shared_motion':list(adapter.motion.parameters())}
    covariance_gradients={}
    for name,parameters in groups.items():
        values=torch.autograd.grad(cov_loss,parameters,retain_graph=True,allow_unused=True)
        assert all(g is None or g.isfinite().all() for g in values)
        covariance_gradients[name]=sum(float(g.abs().sum()) for g in values if g is not None)
        assert covariance_gradients[name]>0,(name,covariance_gradients[name])
    # Isolate the actual rasterizer's covariance backward path from center,
    # opacity and appearance paths; matrix-only autograd is insufficient here.
    bank=scene.dynamic_at(.43)
    bank.get_xyz=bank.get_xyz.detach()
    bank.get_opacity=bank.get_opacity.detach()
    bank.get_features=bank.get_features.detach()
    native_shape_loss=renderer(cam,bank,PIPE,scene.centers.new_zeros(3),0.,0.,0.)['render'].square().mean()
    native_covariance_gradients={}
    for name,parameters in groups.items():
        values=torch.autograd.grad(native_shape_loss,parameters,retain_graph=True,allow_unused=True)
        assert all(g is None or g.isfinite().all() for g in values)
        native_covariance_gradients[name]=sum(float(g.abs().sum()) for g in values if g is not None)
        assert native_covariance_gradients[name]>0,(name,native_covariance_gradients[name])
    with torch.no_grad():
        restored={}; determinant={}
        for time in [0.,.43,1.]:
            points=x[:128]; times=points.new_full((len(points),1),time)
            moved,ld=adapter.motion(points,times,return_logdet=True)
            restored[str(time)]=float((adapter.motion(moved,times,inverse=True)-points).abs().max())
            assert restored[str(time)]<2e-6 and ld.min()>=-1.2 and ld.max()<=1.2
            determinant[str(time)]={'min':float(ld.exp().min()),'max':float(ld.exp().max())}
    write(EXP/'native_preflight.json',{'passed':True,'flags_off_native_rgb_alpha_loss_exact':True,
        'disabled_install_skips_assets_and_keeps_original_field':True,'gradient_max_abs_difference':error,'gradient_scale':scale,
        'same_original_parameter_objects':True,'inactive_original_covariance_field_queries_per_call':calls,
        'new_shared_position_parameters':sum(p.numel() for p in adapter.motion.parameters()),
        'render_physics_center_max_abs_difference_after_updates':mapping,
        'initialization_preserves_CPU_and_CUDA_RNG':True,
        'initial_enabled_rgb_alpha_difference':enabled_error,'dynamic_alpha_uses_precomputed_covariance':True,'round_trip_after_updates_max_abs':restored,
        'native_renderer_covariance_only_gradient_abs_sums':native_covariance_gradients,
        'analytic_determinant_range_after_updates':determinant,
        'canonical_and_shared_motion_gradients_finite_nonzero':True,'inactive_original_grids_and_heads_have_no_gradients':True,'covariance_only_gradient_abs_sums':covariance_gradients,'forced_physics_integration_smoke_updates':20,'forced_physics_calls':2,
        'forced_physics_smoke_is_not_actual16k_warmup':True,'physics_smoke_records':rows})
    print('COVARIANCE_TRANSPORT_NATIVE_PASSED',error,mapping,flush=True)


def audit(smoke=True,target_override=None):
    plan=read(EXP/'protocol.json'); expected=target_override or (plan['smoke_updates'] if smoke else plan['actual_updates'])
    parent=torch.load(INITIAL,map_location='cpu',weights_only=False); results={}
    baseline=[]
    for phase in ['warmup','physics']:
        baseline.extend(json.loads(line)['frame_index'] for line in (ROOT/f'runs/pidg_staged_reconstruction/S3_union/{phase}/formal/sampling.jsonl').read_text().splitlines())
    for arm in (['control','transport'] if smoke else ['transport']):
        directory=WORK/('_smoke' if smoke else 'formal')/arm; target=100 if smoke and arm=='control' else expected
        path=directory/'checkpoint.pt'; assert sha(path)==read(directory/'checkpoint.json')['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False); initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(value,initial['model'][key]) for key,value in parent['model'].items())
        if arm=='control': assert initial['optimizer']==parent['optimizer']
        else:
            assert initial['optimizer']['param_groups'][:-1]==parent['optimizer']['param_groups']
            assert not initial['optimizer']['state']
            assert initial['optimizer']['param_groups'][-1]['name']=='invertible_motion'
        assert state['iteration']==target and state['protocol_sha256']==sha(EXP/'protocol.json')
        assert state['model'].keys()==initial['model'].keys()
        assert set(state['model'])-set(parent['model'])==(set() if arm=='control' else {k for k in state['model'] if k.startswith('networks.1.')})
        rows=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
        assert [row['global_step'] for row in rows]==list(range(1,target+1))
        assert [row['frame_index'] for row in rows]==baseline[:target]
        rng=random.Random(); rng.setstate(parent['rng_python']); stack=list(parent['camera_stack'])
        for row in rows:
            if not stack: stack=rng.sample(range(120),120)
            assert row['frame_index']==stack.pop()
            assert row['covariance_transport']==(arm=='transport') and 'ordinal' not in row and 'local_rigidity' not in row
        assert state['camera_stack']==stack and state['rng_python']==rng.getstate()
        assert torch.equal(parent['rng_torch'],state['rng_torch'])
        assert all(torch.equal(a,b) for a,b in zip(parent['rng_cuda'],state['rng_cuda']))
        assert np.array_equal(parent['rng_numpy'][1],state['rng_numpy'][1])
        assert all(value.isfinite().all() for value in state['model'].values())
        assert all(torch.equal(value,parent['model'][key]) for key,value in state['model'].items() if key.startswith('static.'))
        calls=max(0,target-plan['warmup_updates'])//10
        assert sum('physics' in row for row in rows)==calls
        _,probe,probe_opt,_,_=load_initial()
        if arm=='transport': install(probe,plan,True,probe_opt)
        parameter_names={id(p):name for name,p in probe.named_parameters()}
        names={index:parameter_names[id(p)] for live,stored in zip(probe_opt.param_groups,state['optimizer']['param_groups']) for p,index in zip(live['params'],stored['params'])}
        for group in state['optimizer']['param_groups']:
            for index in group['params']:
                count=calls if group['name']=='PIDG_velocity_stress' else target
                inactive=arm=='transport' and names[index].startswith(('grids.','networks.0.'))
                if count==0 or inactive: assert index not in state['optimizer']['state']
                else:
                    value=state['optimizer']['state'][index]; assert float(value['step'])==count
                    assert all(not torch.is_tensor(v) or v.isfinite().all() for v in value.values())
        del probe,probe_opt
        if smoke and arm=='transport': assert all(read(directory/f'resume_{n:05d}.json')['full_state_restored'] for n in [40,80])
        results[arm]={'actual_optimizer_updates':target,'physics_calls':calls,'checkpoint_sha256':sha(path),
                     'original_initial_state_preserved_and_sampler_exact':True,'static_frozen':True}
    assert read(EXP/'native_preflight.json')['passed'] and sha(INITIAL)==INITIAL_SHA
    output='preflight.json' if smoke else ('warmup_state_audit.json' if target_override else 'training_verification.json')
    write(EXP/output,{'passed':True,'arms':results,'test_RGB_read':False})
    print('COVARIANCE_TRANSPORT_STATE_AUDIT_PASSED',results,flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--native',action='store_true'); parser.add_argument('--audit-smoke',action='store_true')
    parser.add_argument('--audit-formal',action='store_true'); parser.add_argument('--audit-warmup',action='store_true'); args=parser.parse_args()
    if args.native: native()
    elif args.audit_smoke: audit()
    elif args.audit_formal: audit(False)
    elif args.audit_warmup: audit(False,16000)
    else: parser.error('Select operation')


if __name__=='__main__': main()
