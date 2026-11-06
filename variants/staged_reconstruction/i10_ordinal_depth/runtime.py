"""Isolated complete-J6 local pilot; unchanged native PIDG rendering/Adam."""
from contextlib import contextmanager
from copy import deepcopy
import os
import numpy as np
import torch
from ..i10_tracking.io import ROOT,DATA,read,sha
from ..i10_tracking.core import project_world,patch_ncc
from ..i10_track_neighborhood.core import patch_visibility
from ..i10_joint.runtime import load as old_load
from ..rubber_joint import runtime as r
from ..rubber_joint.core import objective as baseline
from ..rubber_visibility.worker import native,colors_for
from ..joint_rgb.core import rng_state,restore_rng
from ..state_recovery.runtime import restore_graphs
from ..train_static import camera
from .core import ranking,sampled_depth,transition

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_ordinal_depth'
REPORT=BASE/'v1';RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_ordinal_depth/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_ordinal_depth/v1'
ORIGINAL_SNAPSHOT=r.snapshot
ARM=CONFIG=BANK=STATE=META=ASSET=None
LAST={}


@contextmanager
def evidence_context(scene):
    rng=deepcopy(rng_state());flags=[p.requires_grad for p in scene.parameters()]
    try:
        with torch.no_grad():yield
    finally:
        for p,value in zip(scene.parameters(),flags):p.requires_grad_(value)
        restore_rng(rng)


def load(arm):
    global ARM,CONFIG,BANK,STATE,META,ASSET
    CONFIG=read(BASE/'protocol.json');assert arm in CONFIG['arms'];ARM=arm
    assert sha(ROOT/CONFIG['parent'])==CONFIG['parent_sha256']
    renderer,scene,opts,_,_=old_load('frozen_static')
    parent=torch.load(ROOT/CONFIG['parent'],map_location='cpu',weights_only=False)
    scene.load_state_dict(parent['model'],strict=True)
    for opt,value in zip(opts,parent['optimizers']):opt.load_state_dict(value)
    assert r.bindings(scene,opts)==parent['bindings'];restore_graphs(scene,parent['physics_graphs'])
    scene.static.requires_grad_(False);scene.physics.requires_grad_(False)
    BANK=read(REPORT/'pair_bank.json');META=read(DATA/'transforms_train.json')
    path=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_tracking/dense/v1/parent_trajectories.npz'
    with np.load(path) as f:ASSET={n:f[n].copy() for n in ['views','k']}
    STATE=deepcopy(read(REPORT/'initial_health.json')['controller']) if (REPORT/'initial_health.json').exists() else dict(calls=0,pairs={},history=[])
    restore_rng(parent)
    r.assert_equivalent(scene.state_dict(),parent['model'],0,0)
    for opt,value in zip(opts,parent['optimizers']):r.assert_equivalent(opt.state_dict(),value,0,0)
    return renderer,scene,opts,None,parent


def measure(renderer,scene):
    result={}
    with evidence_context(scene):
        for index in sorted({q['frame'] for q in BANK['fit']['pairs']}):
            assert not r.stopped(),'User stop request'
            frame=META['frames'][index];cam=camera(frame,META,400)
            pairs=[q for q in BANK['fit']['pairs'] if q['frame']==index]
            assert len({q['row'] for q in pairs})==1,'This fixed local pilot has one admitted track per image'
            pkg=scene.render(renderer,cam,frame['time']);packed=native(renderer,scene,cam,frame['time'],colors_for(scene,[[pairs[0]['row']]]))
            assert all(torch.equal(pkg[k],packed[k]) for k in ['alpha','depth'])
            alpha=pkg['alpha'].squeeze().cpu().numpy();weight=packed['render'][0].cpu().numpy()
            observed=r.frame_target(frame,400).mean(0).cpu().numpy()*255;pred=pkg['render'].mean(0).cpu().numpy()*255
            row=pairs[0]['row'];world=scene.dynamic_at(frame['time']).get_xyz[row].cpu().numpy()
            uv,z=project_world(world[None],ASSET['views'][index:index+1],ASSET['k'])
            for q in pairs:
                point=np.array(q['uv'][0]);error=float(np.linalg.norm(uv[0,0]-point))
                vis=patch_visibility(weight,alpha,point,CONFIG)
                ncc,x,y=patch_ncc(observed,pred,point[None],point[None],2)
                depths,alphas=sampled_depth(pkg,q['uv']);depth_ok=bool(torch.isfinite(depths).all() and (depths>0).all())
                supported=bool((alphas>CONFIG['minimum_rendered_alpha']).all())
                known=depth_ok and supported and vis['known'] and bool(z[0,0]>.01)
                passed=known and vis['passed'] and error<=CONFIG['maximum_projection_error'] and ncc[0]>=CONFIG['ncc_min'] and min(x[0],y[0])>=CONFIG['patch_std_min']
                result[q['id']]=dict(known=known,passed=bool(passed),projection_error=error,
                    visibility=vis,ncc=float(ncc[0]),observed_std=float(x[0]),rendered_std=float(y[0]),
                    endpoint_depths=depths.cpu().tolist(),endpoint_alpha=alphas.cpu().tolist())
    return result


def refresh(renderer,scene,initial=False):
    global STATE
    measured=measure(renderer,scene);before=deepcopy(STATE);states={}
    for q in BANK['fit']['pairs']:
        m=measured[q['id']];s=transition(None if initial else STATE['pairs'][q['id']],m['known'],m['passed'],initial)
        states[q['id']]=dict(**s,measurement=m)
    summary=dict(completed_updates=0 if initial else STATE['calls'],pairs=len(states),
        active=sum(s['status']=='active' for s in states.values()),
        suspended=sum(s['status']=='suspended' for s in states.values()),revoked=sum(s['status']=='revoked' for s in states.values()))
    STATE=dict(calls=0 if initial else STATE['calls'],pairs=states,history=([summary] if initial else STATE['history']+[summary]))
    if not initial:
        directory=RUNS/'refresh'/ARM;directory.mkdir(parents=True,exist_ok=True);path=directory/f'model_{STATE["calls"]:04d}.pt'
        assert not path.exists();torch.save(dict(model=r.cpu_values(scene.state_dict()),model_only_not_resume=True),path)
        r.atomic_json(path.with_suffix('.json'),dict(model_sha256=sha(path),before=before,after=deepcopy(STATE)))
    print('ORDINAL_REFRESH',summary,flush=True)
    return summary


def ordinal(renderer,scene,index,force_fixture=False):
    pairs=[q for q in BANK['fit']['pairs'] if q['frame']==index]
    if not pairs:return scene.centers.sum()*0,dict(total_pairs=0,used_pairs=0,violating_pairs=0)
    frame=META['frames'][index];cam=camera(frame,META,400);pkg=scene.render(renderer,cam,frame['time'])
    uv=[v for q in pairs for v in q['uv']];z,alpha=sampled_depth(pkg,uv);z=z.reshape(-1,2);alpha=alpha.reshape(-1,2)
    valid=torch.isfinite(z).all(1)&(z>0).all(1)&(alpha>CONFIG['minimum_rendered_alpha']).all(1)
    world=scene.dynamic_at(frame['time']).get_xyz
    points=world[torch.tensor([q['row'] for q in pairs],device=world.device)]
    view=torch.as_tensor(ASSET['views'][index],device=world.device,dtype=world.dtype);k=world.new_tensor(ASSET['k'])
    camera_xyz=points@view[:3,:3].T+view[:3,3];xy=camera_xyz@k.T;projection=xy[:,:2]/xy[:,2:3]
    errors=torch.linalg.vector_norm(projection-world.new_tensor([q['uv'][0] for q in pairs]),dim=1)
    valid&=(camera_xyz[:,2]>.01)&(errors<=CONFIG['maximum_projection_error'])
    valid&=torch.tensor([STATE['pairs'][q['id']]['status']=='active' for q in pairs],device=z.device)
    sign=z.new_tensor([q['sign'] for q in pairs]);confidence=z.new_tensor([q['confidence'] for q in pairs])
    if force_fixture:
        # Qualification-only value perturbation, derivative remains native D/alpha.
        delta=sign*(torch.log(z[:,1])-torch.log(z[:,0]));forced=delta-delta.detach()-1
        z=torch.stack([torch.ones_like(forced),torch.exp(sign*forced)],1)
        valid=torch.ones_like(valid);confidence=torch.ones_like(confidence)
    _,aux=ranking(z.sum()*0,z,sign,confidence,valid,CONFIG,True)
    gap=sign*(torch.log(z[:,1])-torch.log(z[:,0]))
    return aux,dict(total_pairs=len(pairs),used_pairs=int(valid.sum()),violating_pairs=int((valid&(gap<np.log(CONFIG['margin_ratio']))).sum()))


def objective(renderer,scene,teacher,cam,t,target,enabled=False):
    assert not enabled
    if STATE['calls']>0 and STATE['calls']%CONFIG['refresh_every']==0:refresh(renderer,scene)
    rgb,pkg,_,_=baseline(renderer,scene,None,cam,t,target,False)
    index=min(range(120),key=lambda i:abs(META['frames'][i]['time']-t));assert abs(META['frames'][index]['time']-t)<1e-9
    aux=rgb*0;counts=dict(total_pairs=0,used_pairs=0,violating_pairs=0)
    if ARM=='ordinal_depth':aux,counts=ordinal(renderer,scene,index)
    STATE['calls']+=1;LAST.clear();LAST.update(weighted_ordinal_loss=float(aux.detach()),index=index,**counts)
    return rgb+aux,pkg,rgb,aux


def snapshot(scene,opts,step,stack,trace):
    state=ORIGINAL_SNAPSHOT(scene,opts,step,stack,trace)
    state.update(stage='L8_local_ordinal_depth',controller=deepcopy(STATE),pair_bank_sha256=sha(REPORT/'pair_bank.json'),
        static_fully_frozen=True,absolute_depth_supervision=False,local_pilot_not_global_promotion=True)
    return state


def restore(path,scene,opts):
    global STATE
    assert sha(path)==read(path.with_suffix('.json'))['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['protocol_sha256']==sha(BASE/'protocol.json') and state['source_sha256']==sha(REPORT/'source_freeze.json')
    scene.load_state_dict(state['model'],strict=True)
    for opt,value in zip(opts,state['optimizers']):opt.load_state_dict(value)
    assert r.bindings(scene,opts)==state['bindings'];restore_graphs(scene,state['physics_graphs']);restore_rng(state)
    STATE=deepcopy(state['controller'])
    r.assert_equivalent(state,snapshot(scene,opts,state['iteration'],state['camera_stack'],state['sample_trace']),0,0)
    return state


def train(arm,qualification=False):
    assert not r.stopped();r.verify();torch.set_num_threads(4)
    if not qualification:assert read(REPORT/'recovery.json')['passed']
    with r.lock('train.lock'):
        renderer,scene,opts,_,parent=load(arm);directory=RUNS/('qualification' if qualification else 'formal')/arm
        directory.mkdir(parents=True,exist_ok=True);assert not (directory/'checkpoint.pt').exists()
        count=CONFIG['qualification_steps'] if qualification else CONFIG['updates'];stack=list(parent['camera_stack']);trace=[];usage=[];completed=0
        r.save(directory/'initial.pt',snapshot(scene,opts,0,stack,trace));r.atomic_json(RUNS/'active.json',dict(status='training',arm=arm,pid=os.getpid(),qualification=qualification))
        for step in range(1,count+1):
            if r.stopped():
                r.save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace));r.atomic_json(directory/'usage.json',usage)
                r.atomic_json(RUNS/'active.json',dict(status='saved_stopped',arm=arm,iteration=completed));return
            index=r.next_index(stack);frame=META['frames'][index];cam=camera(frame,META,1600);target=r.frame_target(frame)
            for opt in opts:opt.zero_grad(set_to_none=True)
            try:
                loss,pkg,rgb,aux=objective(renderer,scene,None,cam,frame['time'],target)
                assert torch.isfinite(loss);loss.backward()
                assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
                r.update(scene,opts,arm);completed=step;trace.append(index);usage.append(dict(**LAST))
            except BaseException as error:
                if r.stopped():
                    stack.append(index)
                    r.save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace));r.atomic_json(directory/'usage.json',usage)
                    r.atomic_json(RUNS/'active.json',dict(status='saved_stopped',arm=arm,iteration=completed));return
                r.atomic_json(directory/'failure.json',dict(error=repr(error),last_completed=completed,partial_step_not_saved=True));raise
            if step==1 or step%25==0 or step==count:
                r.atomic_json(directory/'progress.json',dict(iteration=step,total=count,rgb_loss=float(rgb.detach()),**LAST))
                print('ORDINAL_STEP',arm,step,float(rgb.detach()),float(aux.detach()),LAST['used_pairs'],flush=True)
            del loss,pkg,rgb,aux,target
            if step%CONFIG['checkpoint_every']==0:r.save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace))
        r.save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace));r.atomic_json(directory/'usage.json',usage)
        r.atomic_json(directory/'final_status.json',dict(status='qualification_saved' if qualification else 'trained_pending_evaluation',iteration=completed,pending_review=True));r.verify()


def install(enabled=False):
    if not enabled:return r
    r.BASE=BASE;r.REPORT=REPORT;r.RUNS=RUNS;r.OUT=OUT
    r.load=load;r.objective=objective;r.snapshot=snapshot;r.restore=restore;r.train=train
    return r
