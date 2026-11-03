"""Native disabled-path parity and complete state/optimizer/sampler audits."""
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
from .train import load_initial,load_graph,step_loss


def native():
    torch.set_num_threads(4)
    renderer,scene,opt,initial,base=load_initial()
    plan=read(EXP/'protocol.json'); graph=load_graph(plan,scene.centers.device)
    meta=read(DATA/'transforms_train.json')
    for i,frame in enumerate(meta['frames']): frame['_index']=i
    frame=meta['frames'][0]; rgba=image(frame,800); cam=camera(frame,meta,800)
    gt=torch.tensor(rgba[...,:3]*rgba[...,3:4],device='cuda').permute(2,0,1)
    opt.zero_grad(set_to_none=True)
    expected=reconstruction(scene.render(renderer,cam,frame['time']),gt,rgba,scene,cam)
    expected.backward(); grads={name:None if p.grad is None else p.grad.clone() for name,p in scene.named_parameters()}
    from . import train
    old=train.scene_prior
    def forbidden(*args,**kwargs): raise AssertionError('Disabled geometric branch executed')
    train.scene_prior=forbidden; error,scale=0.,0.
    try:
        for enabled,config in [(False,plan),(True,dict(plan,local_rigidity_enabled=False,relative_rotation_enabled=False))]:
            loss,_=step_loss(scene,renderer,opt,frame,meta,{},1,config,base,enabled,None)
            assert torch.equal(loss,expected); loss.backward()
            for name,p in scene.named_parameters():
                if grads[name] is None: assert p.grad is None
                else:
                    error=max(error,float((p.grad-grads[name]).abs().max())); scale=max(scale,float(grads[name].abs().max()))
    finally: train.scene_prior=old
    assert error<max(1e-7,scale*2e-5)
    cache={}; rows=[]
    # Force the phase solely to test original physics integration. This does
    # not count as an actual16k warmup or formal physics admission.
    for local in range(1,21):
        loss,row=step_loss(scene,renderer,opt,meta['frames'][local-1],meta,cache,16000+local,plan,base,True,graph)
        loss.backward(); assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
        opt.step(); scene.constrain(); rows.append(row)
    assert sum('physics' in row for row in rows)==2
    opt.zero_grad(set_to_none=True)
    terms=old(scene,meta['frames'][20]['time'],21,graph,plan)
    objective=plan['rigid_weight']*terms['rigid']+plan['rotation_weight']*terms['rotation']
    objective.backward()
    gradient_norms={}
    for name,parameters in [('grid',scene.grids.parameters()),('deformation',scene.networks.parameters()),('canonical',[scene.pools[0]._xyz])]:
        values=[p.grad for p in parameters if p.grad is not None]
        norm=float(torch.sqrt(sum((value.square().sum() for value in values))))
        assert np.isfinite(norm) and norm>0; gradient_norms[name]=norm
    assert scene.pools[0]._opacity.grad is None
    assert all(p.grad is None for p in scene.physics.parameters())
    # Compare prior's normalized quaternion convention with PIDG's mapping.
    from utils.general_utils import quaternion_to_matrix
    from .core import quaternion_matrix
    x=scene.pools[0].get_xyz[:100]
    state=scene.deforms[0].step(x,x.new_full((len(x),1),.5)); q=state['d_xyz'][:,:4]
    matrix_error=float((quaternion_matrix(q)-quaternion_to_matrix(q)).abs().max())
    assert matrix_error<1e-6
    write(EXP/'native_preflight.json',{'passed':True,'disabled_loss_exact':True,'two_independent_flags_off_skip_branch':True,
        'gradient_max_abs_difference':error,'gradient_scale':scale,'prior_gradient_norms':gradient_norms,
        'prior_does_not_update_opacity_or_physics_parameters':True,'PIDG_rotation_matrix_max_difference':matrix_error,
        'forced_physics_integration_smoke_updates':20,'forced_physics_calls':2,
        'forced_physics_smoke_is_not_actual16k_warmup':True,'physics_smoke_records':rows})
    print('LOCAL_RIGIDITY_NATIVE_PASSED',error,gradient_norms,flush=True)


def audit(smoke=True,target_override=None):
    plan=read(EXP/'protocol.json'); expected=target_override or (plan['smoke_updates'] if smoke else plan['actual_updates'])
    parent=torch.load(INITIAL,map_location='cpu',weights_only=False); results={}
    baseline=[]
    for phase in ['warmup','physics']:
        baseline.extend(json.loads(line)['frame_index'] for line in (ROOT/f'runs/pidg_staged_reconstruction/S3_union/{phase}/formal/sampling.jsonl').read_text().splitlines())
    for arm in (['control','rigid'] if smoke else ['rigid']):
        directory=WORK/('_smoke' if smoke else 'formal')/arm; target=100 if smoke and arm=='control' else expected
        path=directory/'checkpoint.pt'; assert sha(path)==read(directory/'checkpoint.json')['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False); initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(value,initial['model'][key]) for key,value in parent['model'].items())
        assert initial['optimizer']==parent['optimizer']
        assert state['iteration']==target and state['protocol_sha256']==sha(EXP/'protocol.json')
        assert state['model'].keys()==parent['model'].keys()
        rows=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
        assert [row['global_step'] for row in rows]==list(range(1,target+1))
        assert [row['frame_index'] for row in rows]==baseline[:target]
        rng=random.Random(); rng.setstate(parent['rng_python']); stack=list(parent['camera_stack'])
        for row in rows:
            if not stack: stack=rng.sample(range(120),120)
            assert row['frame_index']==stack.pop()
            assert ('local_rigidity' in row)==(arm=='rigid') and 'ordinal' not in row
        assert state['camera_stack']==stack and state['rng_python']==rng.getstate()
        assert torch.equal(parent['rng_torch'],state['rng_torch'])
        assert all(torch.equal(a,b) for a,b in zip(parent['rng_cuda'],state['rng_cuda']))
        assert np.array_equal(parent['rng_numpy'][1],state['rng_numpy'][1])
        assert all(value.isfinite().all() for value in state['model'].values())
        assert all(torch.equal(value,parent['model'][key]) for key,value in state['model'].items() if key.startswith('static.'))
        calls=max(0,target-plan['warmup_updates'])//10
        assert sum('physics' in row for row in rows)==calls
        for group in state['optimizer']['param_groups']:
            for index in group['params']:
                count=calls if group['name']=='PIDG_velocity_stress' else target
                if count==0: assert index not in state['optimizer']['state']
                else:
                    value=state['optimizer']['state'][index]; assert float(value['step'])==count
                    assert all(not torch.is_tensor(v) or v.isfinite().all() for v in value.values())
        if smoke and arm=='rigid': assert all(read(directory/f'resume_{n:05d}.json')['full_state_restored'] for n in [40,80])
        results[arm]={'actual_optimizer_updates':target,'physics_calls':calls,'checkpoint_sha256':sha(path),
                     'initial_full_state_and_sampler_exact':True,'static_frozen':True}
    assert read(EXP/'native_preflight.json')['passed'] and sha(INITIAL)==INITIAL_SHA
    output='preflight.json' if smoke else ('warmup_state_audit.json' if target_override else 'training_verification.json')
    write(EXP/output,{'passed':True,'arms':results,'test_RGB_read':False})
    print('LOCAL_RIGIDITY_STATE_AUDIT_PASSED',results,flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--native',action='store_true'); parser.add_argument('--audit-smoke',action='store_true')
    parser.add_argument('--audit-formal',action='store_true'); parser.add_argument('--audit-warmup',action='store_true'); args=parser.parse_args()
    if args.native: native()
    elif args.audit_smoke: audit()
    elif args.audit_formal: audit(False)
    elif args.audit_warmup: audit(False,16000)
    else: parser.error('Select operation')


if __name__=='__main__': main()
