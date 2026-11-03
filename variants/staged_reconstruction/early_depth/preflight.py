"""Original initialization, flags-off parity and full-state resume evidence."""
import argparse
import json
import random
import numpy as np
import torch
from ..prepare_data import ROOT, REPORT, DATA, read, write, sha
from ..train_static import camera
from ..stage2.train import image
from ..union.train import reconstruction
from .prepare import EXP, WORK, INITIAL, INITIAL_SHA
from .train import load_initial,step_loss


def native():
    torch.set_num_threads(4)
    renderer,scene,opt,initial,base=load_initial()
    plan=read(EXP/'protocol.json'); meta=read(DATA/'transforms_train.json')
    for i,f in enumerate(meta['frames']): f['_index']=i
    f=meta['frames'][0]; rgba=image(f,800); cam=camera(f,meta,800)
    gt=torch.tensor(rgba[...,:3]*rgba[...,3:4],device='cuda').permute(2,0,1)
    opt.zero_grad(set_to_none=True)
    package=scene.render(renderer,cam,f['time'])
    expected=reconstruction(package,gt,rgba,scene,cam)
    expected.backward()
    grads={n:None if p.grad is None else p.grad.clone() for n,p in scene.named_parameters()}
    # Disabled code must not even call auxiliary rendering or load pair banks.
    from . import train
    old=train.extra_loss
    def forbidden(*args,**kwargs): raise AssertionError('Disabled depth branch executed')
    train.extra_loss=forbidden
    try: loss,record=step_loss(scene,renderer,opt,f,meta,{},1,plan,base,False,{})
    finally: train.extra_loss=old
    assert torch.equal(loss,expected)
    loss.backward(); error,scale=0.,0.
    for n,p in scene.named_parameters():
        if grads[n] is None: assert p.grad is None
        else:
            error=max(error,float((p.grad-grads[n]).abs().max())); scale=max(scale,float(grads[n].abs().max()))
    assert error<max(1e-7,scale*2e-5)
    # Exercise the unchanged physics+early-depth code with actual20 optimizer
    # updates. This is a forced-phase integration smoke, not16k warmup training.
    cache,depth_cache,rows={},{},[]
    for local in range(1,21):
        loss,record=step_loss(scene,renderer,opt,meta['frames'][local-1],meta,cache,16000+local,plan,base,True,depth_cache)
        loss.backward()
        assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
        opt.step(); scene.constrain(); rows.append(record)
    assert sum('physics' in r for r in rows)==2
    write(EXP/'native_preflight.json',{'passed':True,'flags_off_inherited_loss_exact':True,
          'flags_off_auxiliary_branch_not_called':True,'gradient_max_abs_difference':error,
          'gradient_scale':scale,'original_initial_sha256':INITIAL_SHA,
          'forced_physics_integration_smoke_updates':20,'forced_physics_calls':2,
          'forced_physics_smoke_is_not_actual16k_warmup':True,'physics_smoke_records':rows})
    print('EARLY_NATIVE_PASSED',error,flush=True)


def audit(smoke):
    plan=read(EXP/'protocol.json'); expected=plan['smoke_updates'] if smoke else plan['actual_updates']
    arms=['control','ordinal'] if smoke else ['ordinal']
    results={}
    parent=torch.load(INITIAL,map_location='cpu',weights_only=False)
    baseline=[]
    for phase in ['warmup','physics']:
        baseline.extend(json.loads(v)['frame_index'] for v in (ROOT/f'runs/pidg_staged_reconstruction/S3_union/{phase}/formal/sampling.jsonl').read_text().splitlines())
    for arm in arms:
        directory=WORK/('_smoke' if smoke else 'formal')/arm
        target=100 if smoke and arm=='control' else expected
        path=directory/'checkpoint.pt'
        assert sha(path)==read(directory/'checkpoint.json')['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False)
        initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(v,initial['model'][k]) for k,v in parent['model'].items())
        assert initial['optimizer']==parent['optimizer']
        assert state['iteration']==target and state['protocol_sha256']==sha(EXP/'protocol.json')
        rows=[json.loads(v) for v in (directory/'steps.jsonl').read_text().splitlines()]
        assert [r['global_step'] for r in rows]==list(range(1,target+1))
        assert [r['frame_index'] for r in rows]==baseline[:target]
        rng=random.Random(); rng.setstate(parent['rng_python']); stack=list(parent['camera_stack'])
        for r in rows:
            if not stack: stack=rng.sample(range(120),120)
            assert r['frame_index']==stack.pop()
            assert ('ordinal' in r)==(arm=='ordinal')
        assert state['camera_stack']==stack and state['rng_python']==rng.getstate()
        assert torch.equal(parent['rng_torch'],state['rng_torch'])
        assert all(torch.equal(a,b) for a,b in zip(parent['rng_cuda'],state['rng_cuda']))
        assert np.array_equal(parent['rng_numpy'][1],state['rng_numpy'][1])
        assert all(v.isfinite().all() for v in state['model'].values())
        assert all(torch.equal(v,parent['model'][k]) for k,v in state['model'].items() if k.startswith('static.'))
        physical_ids={idx for group in state['optimizer']['param_groups'] if group['name']=='PIDG_velocity_stress' for idx in group['params']}
        physics_calls=max(0,target-plan['warmup_updates'])//10
        assert sum('physics' in r for r in rows)==physics_calls
        for group in state['optimizer']['param_groups']:
            for idx in group['params']:
                count=physics_calls if idx in physical_ids else target
                if count==0: assert idx not in state['optimizer']['state']
                else:
                    value=state['optimizer']['state'][idx]
                    assert float(value['step'])==count,(group['name'],count,value['step'])
                    assert all(not torch.is_tensor(v) or v.isfinite().all() for v in value.values())
        if smoke and arm=='ordinal': assert all(read(directory/f'resume_{n:05d}.json')['full_state_restored'] for n in [40,80])
        results[arm]={'actual_optimizer_updates':target,'physics_calls':physics_calls,'checkpoint_sha256':sha(path),
                     'initial_state_camera_Adam_RNG_exact':True,'same_camera_order_as_original_S3U':True,'static_frozen':True}
    assert read(EXP/'native_preflight.json')['passed'] and sha(INITIAL)==INITIAL_SHA
    write(EXP/('preflight.json' if smoke else 'training_verification.json'),{'passed':True,'arms':results,'test_RGB_read':False})
    print('EARLY_STATE_AUDIT_PASSED',results,flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--native',action='store_true'); p.add_argument('--audit-smoke',action='store_true'); p.add_argument('--audit-formal',action='store_true'); a=p.parse_args()
    if a.native: native()
    elif a.audit_smoke or a.audit_formal: audit(a.audit_smoke)
    else: p.error('Select operation')


if __name__=='__main__': main()
