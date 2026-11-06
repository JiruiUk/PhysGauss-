"""J7 joint appearance/dynamics plus frozen L4 observation-memory algorithm."""
from copy import deepcopy
import numpy as np
import torch
from ..prepare_data import ROOT, DATA, read, sha
from ..colorfree_joint.runtime import load as bootstrap
from ..rubber_joint import runtime as r
from ..rubber_joint.core import configure, objective as base_objective
from ..i10_temporal_memory import runtime as memory
from ..i10_temporal_memory.core import add_memory
from ..joint_rgb.core import restore_rng
from ..state_recovery.runtime import restore_graphs

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_J7_long_memory'
REPORT=BASE/'v1';RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_J7_long_memory/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_J7_long_memory/v1'
ORIGINAL_SNAPSHOT=r.snapshot
LAST={}

def load(arm):
    p=read(BASE/'protocol.json');assert arm in p['arms']
    assert sha(ROOT/p['parent'])==p['parent_sha256']
    renderer,scene,opts,_,_=bootstrap('colorfree_static')
    parent=torch.load(ROOT/p['parent'],map_location='cpu',weights_only=False)
    scene.load_state_dict(parent['model'],strict=True)
    for opt,value in zip(opts,parent['optimizers']):opt.load_state_dict(value)
    assert r.bindings(scene,opts)==parent['bindings'];restore_graphs(scene,parent['physics_graphs'])
    scene.static.requires_grad_(False);scene.physics.requires_grad_(False)
    teacher=configure(scene,True)
    memory.ARM=arm;memory.CONFIG=p;memory.META=read(DATA/'transforms_train.json')
    memory.MEMORIES=[m for m in read(REPORT/'bank.json')['memories'] if m['binding']['accepted']] if (REPORT/'bank.json').exists() else []
    memory.STATE=deepcopy(read(REPORT/'initial_health.json')['controller']) if (REPORT/'initial_health.json').exists() else None
    memory.ASSET=None
    if (ROOT/p['trajectories']).exists():
        with np.load(ROOT/p['trajectories']) as f:memory.ASSET={n:f[n] for n in ['views','k']}
    restore_rng(parent)
    r.assert_equivalent(scene.state_dict(),parent['model'],0,0)
    for opt,value in zip(opts,parent['optimizers']):r.assert_equivalent(opt.state_dict(),value,0,0)
    r.assert_equivalent(scene.static.state_dict(),teacher.state_dict(),0,0)
    return renderer,scene,opts,teacher,parent

def objective(renderer,scene,teacher,cam,t,target,enabled=False):
    base,pkg,rgb,self_loss=base_objective(renderer,scene,teacher,cam,t,target,True)
    if not enabled:return base,pkg,rgb,self_loss
    assert memory.STATE is not None and memory.ARM=='temporal_memory'
    if memory.STATE['calls']>0 and memory.STATE['calls']%memory.CONFIG['refresh_every']==0:memory.refresh(renderer,scene)
    index=min(range(120),key=lambda i:abs(memory.META['frames'][i]['time']-t))
    assert abs(memory.META['frames'][index]['time']-t)<1e-9
    loss,aux,counts=add_memory(base,lambda f:scene.dynamic_at(memory.META['frames'][f]['time']).get_xyz,
        memory.ASSET['views'],memory.ASSET['k'],index,memory.MEMORIES,memory.STATE['memories'],memory.CONFIG,True)
    LAST.update(weighted_memory_loss=float(aux.detach()),**counts)
    return loss,pkg,rgb,self_loss

def step_objective(renderer,scene,teacher,cam,t,target):
    LAST.clear();LAST.update(weighted_memory_loss=0.,memories=0,endpoint_pairs=0,keyframes=[],denominator=0)
    result=objective(renderer,scene,teacher,cam,t,target,memory.ARM=='temporal_memory')
    memory.STATE['calls']+=1
    return result

def snapshot(scene,opts,step,stack,trace):
    state=ORIGINAL_SNAPSHOT(scene,opts,step,stack,trace)
    state.update(stage='J8_J7_observation_memory',controller=deepcopy(memory.STATE),
        evidence_sha256=sha(ROOT/memory.CONFIG['evidence']),bank_sha256=sha(REPORT/'bank.json'),
        color_static_reference=False,static_fully_frozen=False,static_geometry_frozen=True,
        candidate_static_has_no_color_threshold=True,dynamic_historical_color_lineage=True,
        static_spacing=scene.static.spacing,self_depth_teacher_parent_sha256=memory.CONFIG['parent_sha256'])
    return state

def restore(path,scene,opts):
    assert sha(path)==read(path.with_suffix('.json'))['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['protocol_sha256']==sha(BASE/'protocol.json') and state['source_sha256']==sha(REPORT/'source_freeze.json')
    scene.load_state_dict(state['model'],strict=True)
    for opt,value in zip(opts,state['optimizers']):opt.load_state_dict(value)
    assert r.bindings(scene,opts)==state['bindings'];restore_graphs(scene,state['physics_graphs']);restore_rng(state)
    memory.STATE=deepcopy(state['controller'])
    r.assert_equivalent(state,snapshot(scene,opts,state['iteration'],state['camera_stack'],state['sample_trace']),0,0)
    return state

def train(arm,qualification=False):
    from ..train_static import camera
    assert not r.stopped();r.verify();torch.set_num_threads(4)
    if not qualification:assert read(REPORT/'recovery.json')['passed']
    with r.lock('train.lock'):
        renderer,scene,opts,teacher,parent=load(arm)
        directory=RUNS/('qualification' if qualification else 'formal')/arm;directory.mkdir(parents=True,exist_ok=True)
        assert not (directory/'initial.pt').exists(),'No duplicate start or silent resume'
        count=memory.CONFIG['qualification_steps' if qualification else 'updates']
        stack=list(parent['camera_stack']);trace=[];used=[];completed=0
        r.save(directory/'initial.pt',snapshot(scene,opts,0,stack,trace))
        r.atomic_json(RUNS/'active.json',dict(status='training',arm=arm,qualification=qualification))
        try:
            for step in range(1,count+1):
                if r.stopped():raise memory.StopRequested('User stop request')
                index=r.next_index(stack);frame=memory.META['frames'][index]
                for opt in opts:opt.zero_grad(set_to_none=True)
                try:loss,pkg,rgb,self_loss=step_objective(renderer,scene,teacher,camera(frame,memory.META,1600),frame['time'],r.frame_target(frame))
                except memory.StopRequested:stack.append(index);raise
                assert torch.isfinite(loss);loss.backward()
                assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
                r.update(scene,opts,'joint_depth_trust');completed=step;trace.append(index)
                used.append(dict(index=index,self_depth_loss=float(self_loss.detach()),**LAST))
                if step==1 or step%25==0 or step==count:
                    row=dict(iteration=step,total=count,rgb_loss=float(rgb.detach()),**used[-1])
                    r.atomic_json(directory/'progress.json',row);print('J8_STEP',arm,row,flush=True)
                del loss,pkg,rgb,self_loss
                if step%250==0:r.save(directory/'checkpoint.pt',snapshot(scene,opts,step,stack,trace))
            r.save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace));r.verify()
            r.atomic_json(directory/'usage.json',used)
            r.atomic_json(directory/'final_status.json',dict(status='qualification_saved' if qualification else 'trained_pending_evaluation',iteration=completed,pending_review=True))
        except memory.StopRequested:
            r.save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace));r.atomic_json(directory/'usage.json',used)
            r.atomic_json(RUNS/'active.json',dict(status='saved_stopped',arm=arm,iteration=completed));return
        except Exception as error:
            r.atomic_json(directory/'failure.json',dict(error=repr(error),last_completed_in_memory=completed,partial_failed_update_not_saved=True));raise

def recovery_worker(arm,replicate):
    from ..train_static import camera
    torch.set_num_threads(4);r.verify();assert not r.stopped()
    renderer,scene,opts,teacher,_=load(arm);directory=RUNS/'qualification'/arm
    state=restore(directory/'checkpoint.pt',scene,opts)
    before=deepcopy(r.cpu_values(snapshot(scene,opts,state['iteration'],state['camera_stack'],state['sample_trace'])))
    stack=list(state['camera_stack']);index=r.next_index(stack);frame=memory.META['frames'][index]
    for opt in opts:opt.zero_grad(set_to_none=True)
    loss,pkg,_,_=step_objective(renderer,scene,teacher,camera(frame,memory.META,1600),frame['time'],r.frame_target(frame));loss.backward()
    native={k:pkg[k].detach().cpu() for k in ['render','alpha','depth']}
    gradients={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in scene.named_parameters()}
    path=directory/'native_gradients.pt'
    if replicate==0:torch.save(dict(native=native,gradients=gradients),path)
    fixed=torch.load(path,map_location='cpu',weights_only=False)
    r.assert_equivalent(native,fixed['native'],0,0);r.assert_equivalent(gradients,fixed['gradients'],1e-6,1e-5)
    for n,p in scene.named_parameters():p.grad=None if fixed['gradients'][n] is None else fixed['gradients'][n].to(p.device)
    r.update(scene,opts,'joint_depth_trust')
    r.save(directory/f'restored{replicate}.pt',before)
    r.save(directory/f'replay{replicate}.pt',snapshot(scene,opts,21,stack,state['sample_trace']+[index]))
    print('J8_EXACT_STATE_NATIVE_FIXED_ADAM_PASS',arm,replicate,flush=True)

def install(enabled=False):
    if not enabled:return r
    for module in [r,memory]:module.BASE=BASE;module.REPORT=REPORT;module.RUNS=RUNS;module.OUT=OUT
    r.load=load;r.objective=objective;r.snapshot=snapshot;r.restore=restore;r.train=train;r.recovery_worker=recovery_worker
    return r
