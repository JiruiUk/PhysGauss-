"""Full-state RGB-only pilot; immutable inherited physics is explicitly retained."""
from contextlib import contextmanager
from pathlib import Path
import msvcrt
import os
import random
import numpy as np
import torch
from PIL import Image
from ..prepare_data import ROOT, DATA, read, sha
from ..joint_rgb.io import atomic_json
from ..joint_rgb.core import rng_state, restore_rng, assert_equivalent as _assert_equivalent
from ..state_recovery.runtime import graph_state, restore_graphs
from ..train_static import camera, optimizer_for
from .core import configure, objective

BASE = ROOT/'experiments/staged_reconstruction/RubberDuck_joint_pilot'
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/RubberDuck_joint_pilot/v1'
OUT = ROOT/'output/staged_reconstruction/RubberDuck_joint_pilot/v1'


def cpu_values(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_values(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_values(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_values(v) for v in value)
    return value


def assert_equivalent(a,b,atol=0,rtol=0):
    return _assert_equivalent(cpu_values(a),cpu_values(b),atol,rtol)


def stopped():
    return any(p.exists() for p in [ROOT/'stop.request',RUNS/'stop.request',RUNS.parent/'stop.request',RUNS.parent.parent/'stop.request'])


@contextmanager
def lock(name):
    RUNS.mkdir(parents=True,exist_ok=True)
    stream=(RUNS/name).open('a+b');stream.seek(0)
    if not stream.read(1):stream.write(b'0');stream.flush()
    stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
    try: yield
    finally:
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1);stream.close()


def verify():
    for receipt in [BASE/'preregistration.json',REPORT/'source_freeze.json']:
        files=read(receipt);files=files.get('files',files)
        for name,h in files.items():assert sha(ROOT/name)==h,name


def load(arm):
    from ..motion_integrity.prepare import load_parent
    protocol=read(BASE/'protocol.json')
    assert sha(ROOT/protocol['parent'])==protocol['parent_sha256']
    assert sha(ROOT/protocol['static_parent'])==protocol['static_parent_sha256']
    renderer,scene,dynamic_opt,parent,_=load_parent()
    static_parent=torch.load(ROOT/protocol['static_parent'],map_location='cpu',weights_only=False)
    assert_equivalent(static_parent['model'],scene.static.state_dict(),0,0)
    static_opt=optimizer_for(scene.static);static_opt.load_state_dict(static_parent['optimizer'])
    # Every static state/binding is inherited even when it remains frozen.
    assert_equivalent(static_opt.state_dict(),static_parent['optimizer'],0,0)
    teacher=configure(scene,arm=='joint_depth_trust')
    if arm=='frozen_static': teacher=None
    scene.physics.requires_grad_(False)
    restore_rng(parent)
    return renderer,scene,[static_opt,dynamic_opt],teacher,parent


def bindings(scene,opts):
    names={id(p):n for n,p in scene.named_parameters()}
    return [[[names[id(p)] for p in group['params']] for group in opt.param_groups] for opt in opts]


def snapshot(scene,opts,step,stack,trace):
    return {'model':scene.state_dict(),'optimizers':[o.state_dict() for o in opts],
        'bindings':bindings(scene,opts),'physics_graphs':graph_state(scene),
        'iteration':step,'camera_stack':list(stack),'sample_trace':list(trace),**rng_state(),
        'protocol_sha256':sha(BASE/'protocol.json'),'source_sha256':sha(REPORT/'source_freeze.json'),
        'parent_sha256':read(BASE/'protocol.json')['parent_sha256'],
        'stage':'J4_RGB_only','physics_updated':False,'instance_identity_available':False}


def save(path,state):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save(state,tmp);os.replace(tmp,path)
    atomic_json(path.with_suffix('.json'),{'sha256':sha(path),'iteration':state['iteration'],
        'complete_model_two_Adam_rng_sampler_graphs':True})


def restore(path,scene,opts):
    assert sha(path)==read(path.with_suffix('.json'))['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['protocol_sha256']==sha(BASE/'protocol.json')
    assert state['source_sha256']==sha(REPORT/'source_freeze.json')
    scene.load_state_dict(state['model'],strict=True)
    for opt,values in zip(opts,state['optimizers']):opt.load_state_dict(values)
    assert bindings(scene,opts)==state['bindings'];restore_graphs(scene,state['physics_graphs']);restore_rng(state)
    assert_equivalent(state,snapshot(scene,opts,state['iteration'],state['camera_stack'],state['sample_trace']),0,0)
    return state


def frame_target(frame,width=1600):
    path=DATA/(frame['file_path']+'.png')
    if not path.exists():path=ROOT/'data/PIDG/rubber_duck/train'/path.name
    with Image.open(path) as im:
        rgba=np.asarray(im.convert('RGBA').resize((width,width*9//16),Image.Resampling.BILINEAR),dtype=np.float32)/255
    # Dataset transparency only composites black; never supervises any alpha.
    return torch.from_numpy(rgba[...,:3]*rgba[...,3:4]).permute(2,0,1).cuda()


def next_index(stack):
    if not stack:stack.extend(random.sample(range(120),120))
    return stack.pop()


def update(scene,opts,arm):
    if arm=='joint_depth_trust':opts[0].step()
    opts[1].step();scene.constrain()
    if arm=='joint_depth_trust':
        with torch.no_grad():scene.static._opacity.clamp_(-12,12)


def train(arm,qualification=False):
    assert not stopped();verify();torch.set_num_threads(4)
    if not qualification:assert read(REPORT/'recovery.json')['passed']
    with lock('train.lock'):
        renderer,scene,opts,teacher,parent=load(arm)
        directory=RUNS/('qualification' if qualification else 'formal')/arm
        directory.mkdir(parents=True,exist_ok=True)
        count=20 if qualification else read(BASE/'protocol.json')['updates']
        assert not (directory/'checkpoint.pt').exists(),'Do not repeat or silently resume a completed/stopped run'
        stack=list(parent['camera_stack']);trace=[];completed=0
        meta=read(DATA/'transforms_train.json');assert len(meta['frames'])==120
        save(directory/'initial.pt',snapshot(scene,opts,0,stack,trace))
        atomic_json(RUNS/'active.json',{'status':'training','arm':arm,'qualification':qualification,'pid':os.getpid()})
        try:
            for step in range(1,count+1):
                if stopped():
                    save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace))
                    atomic_json(RUNS/'active.json',{'status':'saved_stopped','arm':arm,'iteration':completed});return
                index=next_index(stack);frame=meta['frames'][index];trace.append(index)
                cam=camera(frame,meta,1600);target=frame_target(frame)
                for opt in opts:opt.zero_grad(set_to_none=True)
                loss,package,rgb,self_loss=objective(renderer,scene,teacher,cam,frame['time'],target,arm=='joint_depth_trust')
                assert torch.isfinite(loss);loss.backward()
                assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
                update(scene,opts,arm);completed=step
                if step==1 or step%25==0 or step==count:
                    atomic_json(directory/'progress.json',{'iteration':step,'total':count,'rgb_loss':float(rgb.detach()),
                        'self_depth_loss':float(self_loss.detach()),'sample':index,'pid':os.getpid()})
                    print('J4_STEP',arm,step,float(rgb.detach()),float(self_loss.detach()),flush=True)
                del loss,package,rgb,self_loss,target
                if step%250==0:save(directory/'checkpoint.pt',snapshot(scene,opts,step,stack,trace))
            save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace));verify()
            atomic_json(directory/'final_status.json',{'status':'qualification_saved' if qualification else 'trained_pending_evaluation',
                'iteration':completed,'pending_review':True})
        except Exception as error:
            atomic_json(directory/'failure.json',{'error':repr(error),'last_completed_in_memory':completed,
                'attempted_step':locals().get('step'),'partial_failed_update_not_saved':True});raise


def recovery_worker(arm,replicate):
    torch.set_num_threads(4);assert not stopped();verify()
    renderer,scene,opts,teacher,parent=load(arm);directory=RUNS/'qualification'/arm
    state=restore(directory/'checkpoint.pt',scene,opts)
    pre=snapshot(scene,opts,state['iteration'],state['camera_stack'],state['sample_trace'])
    stack=list(state['camera_stack']);index=next_index(stack);meta=read(DATA/'transforms_train.json')
    frame=meta['frames'][index];cam=camera(frame,meta,1600);target=frame_target(frame)
    for o in opts:o.zero_grad(set_to_none=True)
    loss,package,_,_=objective(renderer,scene,teacher,cam,frame['time'],target,arm=='joint_depth_trust')
    loss.backward()
    grads={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in scene.named_parameters()}
    native={k:package[k].detach().cpu() for k in ['render','alpha','depth']}
    reference=directory/'native_gradients.pt'
    if replicate==0:torch.save({'native':native,'gradients':grads},reference)
    fixed=torch.load(reference,map_location='cpu',weights_only=False)
    assert_equivalent(native,fixed['native'],0,0)
    assert_equivalent(grads,fixed['gradients'],1e-6,1e-5)
    for n,p in scene.named_parameters():
        p.grad=None if fixed['gradients'][n] is None else fixed['gradients'][n].to(p.device)
    update(scene,opts,arm)
    save(directory/f'replay{replicate}.pt',snapshot(scene,opts,21,stack,state['sample_trace']+[index]))
    save(directory/f'restored{replicate}.pt',pre)
    print('J4_RECOVERY_NATIVE_AND_FIXED_ADAM_PASS',arm,replicate,flush=True)
