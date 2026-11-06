"""Isolated temporal-memory adapter; frozen PIDG renderer and weights untouched."""
from contextlib import contextmanager
from copy import deepcopy
import numpy as np
import torch
from ..i10_tracking.io import ROOT, DATA, read, sha
from ..i10_tracking.core import project_world, patch_ncc
from ..i10_joint.runtime import load as old_load
from ..rubber_joint import runtime as r
from ..rubber_joint.core import objective as rgb_objective
from ..rubber_visibility.worker import native, colors_for
from ..i10_track_neighborhood.core import patch_visibility
from ..i10_revocable_surface.core import transition
from ..joint_rgb.core import rng_state, restore_rng
from ..state_recovery.runtime import restore_graphs
from ..train_static import camera
from .core import measurement, add_memory

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_temporal_memory'
REPORT=BASE/'v1';RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_temporal_memory/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_temporal_memory/v1'
ORIGINAL_SNAPSHOT=r.snapshot
ARM=CONFIG=MEMORIES=STATE=META=ASSET=None
LAST={}

class StopRequested(RuntimeError):pass

@contextmanager
def evidence_context(scene):
    rng=deepcopy(rng_state()); flags=[p.requires_grad for p in scene.parameters()]
    try:
        with torch.no_grad():yield
    finally:
        for p,value in zip(scene.parameters(), flags):p.requires_grad_(value)
        restore_rng(rng)

def load(arm):
    global ARM,CONFIG,MEMORIES,STATE,META,ASSET
    CONFIG=read(BASE/'protocol.json'); assert arm in CONFIG['arms'];ARM=arm
    assert sha(ROOT/CONFIG['parent'])==CONFIG['parent_sha256']
    renderer,scene,opts,_,_=old_load('frozen_static')
    parent=torch.load(ROOT/CONFIG['parent'],map_location='cpu',weights_only=False)
    scene.load_state_dict(parent['model'],strict=True)
    for opt,value in zip(opts,parent['optimizers']):opt.load_state_dict(value)
    assert r.bindings(scene,opts)==parent['bindings'];restore_graphs(scene,parent['physics_graphs'])
    scene.static.requires_grad_(False);scene.physics.requires_grad_(False)
    MEMORIES=[m for m in read(REPORT/'bank.json')['memories'] if m['binding']['accepted']]
    STATE=deepcopy(read(REPORT/'initial_health.json')['controller']) if (REPORT/'initial_health.json').exists() else None
    META=read(DATA/'transforms_train.json')
    with np.load(ROOT/CONFIG['trajectories']) as f:ASSET={n:f[n] for n in ['views','k']}
    restore_rng(parent)
    r.assert_equivalent(scene.state_dict(),parent['model'],0,0)
    for opt,value in zip(opts,parent['optimizers']):r.assert_equivalent(opt.state_dict(),value,0,0)
    return renderer,scene,opts,None,parent

def measure(renderer,scene):
    visibility=[[None]*len(m['frames']) for m in MEMORIES]
    appearance=[[None]*len(m['frames']) for m in MEMORIES]
    centers=[np.zeros((len(m['frames']),3),np.float64) for m in MEMORIES]
    with evidence_context(scene):
        for index,frame in enumerate(META['frames']):
            if r.stopped():raise StopRequested('User stop request')
            active=[(j,m['frames'].index(index)) for j,m in enumerate(MEMORIES) if index in m['frames']]
            if not active:continue
            world=scene.dynamic_at(frame['time']).get_xyz.cpu().numpy()
            cam=camera(frame,META,400);raw=scene.render(renderer,cam,frame['time'])
            observed=r.frame_target(frame,400).mean(0).cpu().numpy()*255
            rendered=raw['render'].mean(0).cpu().numpy()*255
            for j,t in active:
                centers[j][t]=world[MEMORIES[j]['binding']['row']]
                uv=np.array([MEMORIES[j]['uv'][t]],np.float32)
                ncc,x,y=patch_ncc(observed,rendered,uv,uv,CONFIG['visibility_patch_radius'])
                appearance[j][t]=dict(ncc=float(ncc[0]),observed_std=float(x[0]),rendered_std=float(y[0]))
            rows=sorted({MEMORIES[j]['binding']['row'] for j,_ in active})
            for begin in range(0,len(rows),3):
                group=rows[begin:begin+3]
                packed=native(renderer,scene,cam,frame['time'],colors_for(scene,[[row] for row in group]))
                assert all(torch.equal(raw[k],packed[k]) for k in ['alpha','depth'])
                alpha=packed['alpha'].squeeze().cpu().numpy()
                for channel,row in enumerate(group):
                    weight=packed['render'][channel].cpu().numpy()
                    for j,t in active:
                        if MEMORIES[j]['binding']['row']==row:
                            visibility[j][t]=patch_visibility(weight,alpha,MEMORIES[j]['uv'][t],CONFIG)
            del raw,packed
    measures=[]
    for j,m in enumerate(MEMORIES):
        uv,z=project_world(centers[j][:,None],ASSET['views'][m['frames']],ASSET['k'])
        measures.append(measurement(m,uv[:,0],z[:,0],visibility[j],appearance[j],CONFIG))
    return measures

def refresh(renderer,scene,initial=False):
    global STATE
    completed=0 if initial else STATE['calls'];before=deepcopy(STATE)
    if not initial:
        directory=RUNS/'native_refresh'/ARM;directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'model_{completed:04d}.pt';assert not path.exists()
        torch.save(dict(model=r.cpu_values(scene.state_dict()),completed_updates=completed,model_only_not_resume=True),path)
    measures=measure(renderer,scene)
    states=[transition(None if initial else STATE['memories'][i],m,CONFIG,initial) for i,m in enumerate(measures)]
    active=[m for m,s in zip(MEMORIES,states) if s['status']=='active']
    summary=dict(completed_updates=completed,total_memories=len(MEMORIES),active_memories=len(active),
        active_rows=len({m['binding']['row'] for m in active}),
        active_observations=sum(len(m['frames']) for m in active),
        active_frame_thirds=[sum(sum(f//40==b for f in m['frames']) for m in active) for b in range(3)],
        suspended=sum(s['status']=='suspended' for s in states),revoked=sum(s['status']=='revoked' for s in states))
    STATE=dict(calls=completed,memories=states,history=[summary] if initial else STATE['history']+[summary])
    if not initial:
        r.atomic_json(path.with_suffix('.json'),dict(model_path=path.relative_to(ROOT).as_posix(),model_sha256=sha(path),
            controller_before=before,controller_after=deepcopy(STATE),summary=summary))
    print('L4_MEMORY_REFRESH',summary,flush=True)
    return summary

def objective(renderer,scene,teacher,cam,t,target,enabled=False):
    assert not enabled and STATE is not None
    if ARM=='temporal_memory' and STATE['calls']>0 and STATE['calls']%CONFIG['refresh_every']==0:
        refresh(renderer,scene)
    rgb,package,_,_=rgb_objective(renderer,scene,None,cam,t,target,False)
    index=min(range(120),key=lambda i:abs(META['frames'][i]['time']-t));assert abs(META['frames'][index]['time']-t)<1e-9
    world_at=lambda f:scene.dynamic_at(META['frames'][f]['time']).get_xyz
    loss,aux,counts=add_memory(rgb,world_at,ASSET['views'],ASSET['k'],index,MEMORIES,STATE['memories'],CONFIG,ARM=='temporal_memory')
    STATE['calls']+=1;LAST.update(weighted_memory_loss=float(aux.detach()),**counts)
    return loss,package,rgb,aux

def snapshot(scene,opts,step,stack,trace):
    state=ORIGINAL_SNAPSHOT(scene,opts,step,stack,trace)
    state.update(stage='L4_temporal_memory_RGB',controller=deepcopy(STATE),
        evidence_sha256=sha(ROOT/CONFIG['evidence']),bank_sha256=sha(REPORT/'bank.json'),
        color_static_reference=False,old_I10_target_neutralization=False,static_fully_frozen=True,
        memory_is_not_independent_3D_GT=True)
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
        renderer,scene,opts,teacher,parent=load(arm)
        directory=RUNS/('qualification' if qualification else 'formal')/arm;directory.mkdir(parents=True,exist_ok=True)
        count=CONFIG['qualification_steps'] if qualification else CONFIG['updates'];assert not (directory/'checkpoint.pt').exists()
        stack=list(parent['camera_stack']);trace=[];used=[];completed=0
        r.save(directory/'initial.pt',snapshot(scene,opts,0,stack,trace))
        r.atomic_json(RUNS/'active.json',dict(status='training',arm=arm,qualification=qualification))
        try:
            for step in range(1,count+1):
                if r.stopped():raise StopRequested('User stop request')
                index=r.next_index(stack);frame=META['frames'][index];cam=camera(frame,META,1600);target=r.frame_target(frame)
                for opt in opts:opt.zero_grad(set_to_none=True)
                try:loss,package,rgb,aux=objective(renderer,scene,teacher,cam,frame['time'],target)
                except StopRequested:
                    stack.append(index);raise
                assert torch.isfinite(loss);loss.backward()
                assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
                r.update(scene,opts,arm);completed=step;trace.append(index);used.append(dict(index=index,**LAST))
                if step==1 or step%25==0 or step==count:
                    r.atomic_json(directory/'progress.json',dict(iteration=step,total=count,rgb_loss=float(rgb.detach()),**LAST))
                    print('L4_STEP',arm,step,float(rgb.detach()),float(aux.detach()),LAST['endpoint_pairs'],flush=True)
                del loss,package,rgb,aux,target
                if step%250==0:r.save(directory/'checkpoint.pt',snapshot(scene,opts,step,stack,trace))
            r.save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace));r.verify()
            r.atomic_json(directory/'usage.json',used)
            r.atomic_json(directory/'final_status.json',dict(status='qualification_saved' if qualification else 'trained_pending_evaluation',iteration=completed,pending_review=True))
        except StopRequested:
            r.save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace));r.atomic_json(directory/'usage.json',used)
            r.atomic_json(RUNS/'active.json',dict(status='saved_stopped',arm=arm,iteration=completed));return
        except Exception as error:
            r.atomic_json(directory/'failure.json',dict(error=repr(error),last_completed_in_memory=completed,partial_failed_update_not_saved=True));raise

def install(enabled=False):
    if not enabled:return r
    r.BASE=BASE;r.REPORT=REPORT;r.RUNS=RUNS;r.OUT=OUT
    r.load=load;r.objective=objective;r.snapshot=snapshot;r.restore=restore;r.train=train
    return r
