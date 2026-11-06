"""Full-state bank replacement; frozen PIDG/J6 sources are reused unchanged."""
from copy import deepcopy
from pathlib import Path
import torch
from ..prepare_data import ROOT, DATA, read, sha
from ..rubber_joint import runtime as r
from ..rubber_joint.core import configure, objective
from ..joint_rgb.core import restore_rng
from ..state_recovery.runtime import graph_state, restore_graphs
from .core import validate_payload, replace_static

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_colorfree_joint'
REPORT=BASE/'v1'
RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_colorfree_joint/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_colorfree_joint/v1'
ORIGINAL_SNAPSHOT=r.snapshot


def load(arm):
    from ..motion_integrity.prepare import load_parent
    from ..adaptive_static.model import AdaptiveGaussian
    from ..train_static import optimizer_for
    p=read(BASE/'protocol.json')
    assert arm in p['arms']
    for key in ['parent','static_parent']:assert sha(ROOT/p[key])==p[key+'_sha256']
    renderer,scene,dynamic_opt,_,_=load_parent()
    j6=torch.load(ROOT/p['parent'],map_location='cpu',weights_only=False)
    scene.load_state_dict(j6['model'],strict=True)
    dynamic_opt.load_state_dict(j6['optimizers'][1]);restore_graphs(scene,j6['physics_graphs'])
    static_state=j6['optimizers'][0]
    if arm=='colorfree_static':
        source=torch.load(ROOT/p['static_parent'],map_location='cpu',weights_only=False)
        assert source['iteration']==18000 and source['static_only'] and not source['removed_ids']
        assert validate_payload(source['model'])==p['candidate_static_points']
        bank=AdaptiveGaussian(ROOT/p['static_initialization']);bank.load_payload(source['model'])
        replace_static(scene,bank,True);static_state=source['optimizer']
        r.assert_equivalent(scene.static.state_dict(),source['model'],0,0)
    else:
        assert replace_static(scene) is scene
    assert len(scene.static.get_xyz)==p['candidate_static_points' if arm=='colorfree_static' else 'old_static_points']
    assert len(scene.pools[0].get_xyz)==p['dynamic_points']
    for name,value in scene.state_dict().items():
        if not name.startswith('static.'):assert torch.equal(value.cpu(),j6['model'][name]),name
    static_opt=optimizer_for(scene.static);static_opt.load_state_dict(static_state)
    r.assert_equivalent(static_opt.state_dict(),static_state,0,0)
    r.assert_equivalent(dynamic_opt.state_dict(),j6['optimizers'][1],0,0)
    scene.static.requires_grad_(False);scene.physics.requires_grad_(False)
    teacher=configure(scene,True)
    scene._fusion_arm=arm
    restore_rng(j6)
    return renderer,scene,[static_opt,dynamic_opt],teacher,j6


def snapshot(scene,opts,step,stack,trace):
    result=ORIGINAL_SNAPSHOT(scene,opts,step,stack,trace)
    result.update(stage='J7_colorfree_static_fusion_RGB_only',fusion_arm=scene._fusion_arm,
                  candidate_static_has_no_color_threshold=True,dynamic_historical_color_lineage=True,
                  static_spacing=scene.static.spacing,static_source=read(BASE/'protocol.json')['static_parent']
                  if scene._fusion_arm=='colorfree_static' else read(BASE/'protocol.json')['parent'])
    return result


def train(arm,qualification=False):
    assert not r.stopped();r.verify();torch.set_num_threads(4)
    if not qualification:assert read(REPORT/'recovery.json')['passed']
    with r.lock('train.lock'):
        renderer,scene,opts,teacher,parent=load(arm)
        directory=RUNS/('qualification' if qualification else 'formal')/arm
        directory.mkdir(parents=True,exist_ok=True)
        assert not (directory/'initial.pt').exists(),'No duplicate start or silent resume'
        count=read(BASE/'protocol.json')['qualification_steps' if qualification else 'updates']
        stack=list(parent['camera_stack']);trace=[];completed=0
        r.save(directory/'initial.pt',snapshot(scene,opts,0,stack,trace))
        meta=read(DATA/'transforms_train.json')
        r.atomic_json(RUNS/'active.json',dict(status='training',arm=arm,qualification=qualification))
        try:
            from ..train_static import camera
            for step in range(1,count+1):
                if r.stopped():
                    r.save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace))
                    r.atomic_json(RUNS/'active.json',dict(status='saved_stopped',arm=arm,iteration=completed));return
                index=r.next_index(stack);trace.append(index);frame=meta['frames'][index]
                cam=camera(frame,meta,1600);target=r.frame_target(frame)
                for opt in opts:opt.zero_grad(set_to_none=True)
                loss,package,rgb,self_loss=objective(renderer,scene,teacher,cam,frame['time'],target,True)
                assert torch.isfinite(loss);loss.backward()
                assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
                r.update(scene,opts,'joint_depth_trust');completed=step
                if step==1 or step%25==0 or step==count:
                    row=dict(iteration=step,total=count,arm=arm,rgb_loss=float(rgb.detach()),self_depth_loss=float(self_loss.detach()),sample=index)
                    r.atomic_json(directory/'progress.json',row);print('J7_STEP',row,flush=True)
                del loss,package,rgb,self_loss,target
                if step%250==0:r.save(directory/'checkpoint.pt',snapshot(scene,opts,step,stack,trace))
            r.save(directory/'checkpoint.pt',snapshot(scene,opts,completed,stack,trace));r.verify()
            r.atomic_json(directory/'final_status.json',dict(status='qualified_saved' if qualification else 'trained_pending_evaluation',iteration=completed))
        except Exception as error:
            r.atomic_json(directory/'failure.json',dict(error=repr(error),last_completed_in_memory=completed,attempted_step=locals().get('step')));raise


def recovery_worker(arm,replicate):
    from ..train_static import camera
    torch.set_num_threads(4);r.verify();assert not r.stopped()
    renderer,scene,opts,teacher,_=load(arm);directory=RUNS/'qualification'/arm
    state=r.restore(directory/'checkpoint.pt',scene,opts)
    before=deepcopy(r.cpu_values(snapshot(scene,opts,state['iteration'],state['camera_stack'],state['sample_trace'])))
    stack=list(state['camera_stack']);index=r.next_index(stack)
    meta=read(DATA/'transforms_train.json');frame=meta['frames'][index]
    for opt in opts:opt.zero_grad(set_to_none=True)
    loss,package,_,_=objective(renderer,scene,teacher,camera(frame,meta,1600),frame['time'],r.frame_target(frame),True)
    loss.backward()
    native={k:package[k].detach().cpu() for k in ['render','alpha','depth']}
    gradients={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in scene.named_parameters()}
    path=directory/'native_gradients.pt'
    if replicate==0:torch.save(dict(native=native,gradients=gradients),path)
    fixed=torch.load(path,map_location='cpu',weights_only=False)
    r.assert_equivalent(native,fixed['native'],0,0)
    r.assert_equivalent(gradients,fixed['gradients'],1e-6,1e-5)
    for n,p in scene.named_parameters():p.grad=None if fixed['gradients'][n] is None else fixed['gradients'][n].to(p.device)
    r.update(scene,opts,'joint_depth_trust')
    r.save(directory/f'restored{replicate}.pt',before)
    r.save(directory/f'replay{replicate}.pt',snapshot(scene,opts,21,stack,state['sample_trace']+[index]))
    print('J7_EXACT_STATE_NATIVE_FIXED_GRADIENT_ADAM_PASS',arm,replicate,flush=True)


def install(enabled=False):
    if not enabled:return r
    r.BASE=BASE;r.REPORT=REPORT;r.RUNS=RUNS;r.OUT=OUT
    r.load=load;r.snapshot=snapshot;r.train=train;r.recovery_worker=recovery_worker
    return r
