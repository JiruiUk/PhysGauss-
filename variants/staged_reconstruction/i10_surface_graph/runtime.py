"""Isolated PIDG adapter, exact controller checkpoints, native RGB evidence."""
from contextlib import contextmanager
from copy import deepcopy
import numpy as np
import torch
from ..i10_tracking.io import ROOT,DATA,read,sha
from ..i10_tracking.core import project_world,patch_ncc
from ..i10_joint.runtime import load as old_load
from ..rubber_joint import runtime as r
from ..rubber_joint.core import objective as rgb_objective
from ..rubber_visibility.worker import native,colors_for
from ..i10_track_neighborhood.core import patch_visibility
from ..joint_rgb.core import rng_state,restore_rng
from ..state_recovery.runtime import restore_graphs
from ..train_static import camera
from .core import transition,patch_measurement,edge_measurement,add_constraints

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_surface_graph'
REPORT=BASE/'v1';RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_surface_graph/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_I10_surface_graph/v1'
ORIGINAL_SNAPSHOT=r.snapshot
ARM=CONFIG=PATCHES=EDGES=STATE=META=ASSET=None
LAST={}


class StopRequested(RuntimeError):pass


@contextmanager
def evidence_context(scene):
    rng=deepcopy(rng_state());flags=[p.requires_grad for p in scene.parameters()]
    try:
        with torch.no_grad():yield
    finally:
        for p,value in zip(scene.parameters(),flags):p.requires_grad_(value)
        restore_rng(rng)


def load(arm):
    global ARM,CONFIG,PATCHES,EDGES,STATE,META,ASSET
    assert arm in ['rgb_control','surface_graph'];ARM=arm;CONFIG=read(BASE/'protocol.json')
    assert sha(ROOT/CONFIG['parent'])==CONFIG['parent_sha256']
    renderer,scene,opts,_,_=old_load('frozen_static')
    parent=torch.load(ROOT/CONFIG['parent'],map_location='cpu',weights_only=False)
    scene.load_state_dict(parent['model'],strict=True)
    for opt,value in zip(opts,parent['optimizers']):opt.load_state_dict(value)
    assert r.bindings(scene,opts)==parent['bindings'];restore_graphs(scene,parent['physics_graphs'])
    scene.static.requires_grad_(False);scene.physics.requires_grad_(False)
    hypotheses=read(REPORT/'hypotheses.json');PATCHES=hypotheses['patches'];EDGES=hypotheses['edges']
    STATE=deepcopy(read(REPORT/'initial_health.json')['controller']) if (REPORT/'initial_health.json').exists() else None
    META=read(DATA/'transforms_train.json')
    with np.load(ROOT/CONFIG['trajectories']) as f:ASSET={n:f[n] for n in ['views','k']}
    restore_rng(parent)
    r.assert_equivalent(scene.state_dict(),parent['model'],0,0)
    for opt,value in zip(opts,parent['optimizers']):r.assert_equivalent(opt.state_dict(),value,0,0)
    return renderer,scene,opts,None,parent


def measure(renderer,scene):
    """Read-only native full-scene visibility and raw RGB, never val/test."""
    observations=[[None]*12 for _ in PATCHES];appearances=[[None]*12 for _ in PATCHES]
    centers=np.zeros((len(PATCHES),12,3),np.float64)
    with evidence_context(scene):
        for index,frame in enumerate(META['frames']):
            if r.stopped():raise StopRequested('Save completed state; do not resume automatically')
            active=[(j,index-q['start']) for j,q in enumerate(PATCHES) if q['start']<=index<q['start']+12]
            if not active:continue
            world=scene.dynamic_at(frame['time']).get_xyz.cpu().numpy()
            for j,t in active:centers[j,t]=np.einsum('nc,n->c',world[PATCHES[j]['rows']],PATCHES[j]['weights'])
            cam=camera(frame,META,400);raw=scene.render(renderer,cam,frame['time'])
            observed=r.frame_target(frame,400).mean(0).cpu().numpy()*255
            rendered=raw['render'].mean(0).cpu().numpy()*255
            for j,t in active:
                uv=np.array([PATCHES[j]['uv'][t]],np.float32)
                ncc,a,b=patch_ncc(observed,rendered,uv,uv,CONFIG['visibility_patch_radius'])
                appearances[j][t]=dict(ncc=float(ncc[0]),observed_std=float(a[0]),rendered_std=float(b[0]))
            for start in range(0,len(active),3):
                group=active[start:start+3]
                packed=native(renderer,scene,cam,frame['time'],colors_for(scene,[PATCHES[j]['rows'] for j,_ in group]))
                assert all(torch.equal(raw[k],packed[k]) for k in ['alpha','depth'])
                alpha=packed['alpha'].squeeze().cpu().numpy()
                for channel,(j,t) in enumerate(group):
                    observations[j][t]=patch_visibility(packed['render'][channel].cpu().numpy(),alpha,PATCHES[j]['uv'][t],CONFIG)
            del raw,packed
    measures=[]
    for j,patch in enumerate(PATCHES):
        start=patch['start'];pred,depth=project_world(centers[j,:,None],ASSET['views'][start:start+12],ASSET['k'])
        measures.append(patch_measurement(pred[:,0],depth[:,0],observations[j],appearances[j],patch['uv'],CONFIG))
    return centers,measures


def refresh(renderer,scene,initial=False):
    global STATE
    centers,measurements=measure(renderer,scene)
    previous=STATE
    states=[]
    for i,m in enumerate(measurements):
        owner=PATCHES[i].get('owner')
        if owner is not None and states[owner]['status']!='active':
            m=dict(m,passed=False,kind='unknown',confidence=0.,root_not_active=True)
        states.append(transition(None if initial else previous['patches'][i],m,CONFIG,initial))
    edges=[]
    for i,edge in enumerate(EDGES):
        left,right=edge['i'],edge['j'];lengths=np.linalg.norm(centers[right]-centers[left],axis=-1)
        m=edge_measurement(lengths,states[left],states[right],CONFIG)
        edges.append(transition(None if initial else previous['edges'][i],m,CONFIG,initial))
    summary=dict(completed_updates=0 if initial else previous['calls'],patches=len(states),edges=len(edges),
        active_patches=sum(s['status']=='active' for s in states),active_edges=sum(s['status']=='active' for s in edges),
        suspended_patches=sum(s['status']=='suspended' for s in states),revoked_patches=sum(s['status']=='revoked' for s in states),
        unknown_measurements=sum(m['kind']=='unknown' for m in measurements),
        active_roots=sum(s['status']=='active' for q,s in zip(PATCHES,states) if q['is_anchor']),
        active_neighbor_rows=len({q['rows'][0] for q,s in zip(PATCHES,states) if not q['is_anchor'] and s['status']=='active'}))
    if not initial:
        summary['patch_transitions']=[dict(id=PATCHES[i]['id'],old=a['status'],new=b['status'])
            for i,(a,b) in enumerate(zip(previous['patches'],states)) if a['status']!=b['status']]
        summary['edge_transitions']=[dict(index=i,old=a['status'],new=b['status'])
            for i,(a,b) in enumerate(zip(previous['edges'],edges)) if a['status']!=b['status']]
    STATE=dict(calls=0 if initial else previous['calls'],patches=states,edges=edges,
        history=([summary] if initial else previous['history']+[summary]))
    print('L3_EVIDENCE_REFRESH',summary['completed_updates'],summary['active_patches'],summary['active_edges'],flush=True)
    return summary


def objective(renderer,scene,teacher,cam,t,target,enabled=False):
    assert not enabled and STATE is not None
    if ARM=='surface_graph' and STATE['calls']>0 and STATE['calls']%CONFIG['refresh_every']==0:
        refresh(renderer,scene)
    rgb,package,_,_=rgb_objective(renderer,scene,None,cam,t,target,False)
    index=min(range(120),key=lambda i:abs(META['frames'][i]['time']-t));assert abs(META['frames'][index]['time']-t)<1e-9
    world=scene.dynamic_at(t).get_xyz
    k=world.new_tensor(ASSET['k']);view=cam.world_view_transform.T
    loss,track,local,counts=add_constraints(rgb,world,view,k,index,PATCHES,EDGES,STATE['patches'],STATE['edges'],CONFIG,ARM=='surface_graph')
    STATE['calls']+=1
    LAST.update(weighted_track_loss=float(track.detach()),weighted_local_loss=float(local.detach()),**counts,
                active_patches=sum(s['status']=='active' for s in STATE['patches']),refreshes=len(STATE['history']))
    return loss,package,rgb,track+local


def snapshot(scene,opts,step,stack,trace):
    state=ORIGINAL_SNAPSHOT(scene,opts,step,stack,trace)
    state.update(stage='L3_surface_graph_RGB',controller=deepcopy(STATE),
        evidence_sha256=sha(ROOT/CONFIG['evidence']),hypotheses_sha256=sha(REPORT/'hypotheses.json'),
        color_static_reference=False,old_I10_target_neutralization=False,static_fully_frozen=True,
        original_global_coverage_not_promoted=True,surface_hypotheses_are_not_3D_GT=True)
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
        count=20 if qualification else CONFIG['updates'];assert not (directory/'checkpoint.pt').exists()
        stack=list(parent['camera_stack']);trace=[];completed=0;used=[]
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
                    print('L3_STEP',arm,step,float(rgb.detach()),float(aux.detach()),LAST['tracks'],LAST['edges'],flush=True)
                del loss,package,rgb,aux,target
                if step%250==0:r.save(directory/'checkpoint.pt',snapshot(scene,opts,step,stack,trace))
            r.save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace));r.verify()
            r.atomic_json(directory/'usage.json',used)
            r.atomic_json(directory/'final_status.json',dict(status='qualification_saved' if qualification else 'trained_pending_evaluation',iteration=completed,pending_review=True))
        except StopRequested:
            r.save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace))
            r.atomic_json(directory/'usage.json',used)
            r.atomic_json(RUNS/'active.json',dict(status='saved_stopped',arm=arm,iteration=completed));return
        except Exception as error:
            r.atomic_json(directory/'failure.json',dict(error=repr(error),last_completed_in_memory=completed,partial_failed_update_not_saved=True));raise


def atomic(path,value):
    if isinstance(value,dict) and 'self_depth_loss' in value:
        value=dict(value);value['weighted_auxiliary_loss']=value.pop('self_depth_loss');value.update(LAST)
    r.ATOMIC_ORIGINAL(path,value)


def install(enabled=False):
    if not enabled:return r
    r.BASE=BASE;r.REPORT=REPORT;r.RUNS=RUNS;r.OUT=OUT
    r.load=load;r.objective=objective;r.snapshot=snapshot;r.restore=restore;r.train=train
    return r
