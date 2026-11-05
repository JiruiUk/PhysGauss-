"""Artifact-local adapter: untouched native renderer/deformation and full states."""
from copy import deepcopy
import torch
from ..i10_tracking.io import ROOT,DATA,read,sha
from ..rubber_joint import runtime as r
from ..rubber_joint.core import objective as rgb_objective
from ..i10_joint.runtime import load as old_load
from ..joint_rgb.core import restore_rng
from ..state_recovery.runtime import restore_graphs
from .core import observations,add_tracks

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_tracking/training'
REPORT=BASE/'v1'
RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_tracking/training/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_I10_tracking/training/v1'
SNAPSHOT=r.snapshot
ATOMIC=r.atomic_json
ARM=None
OBS=None
TIMES=None


def load(arm):
    global ARM,OBS,TIMES
    ARM=arm;p=read(BASE/'protocol.json');assert sha(ROOT/p['parent'])==p['parent_sha256']
    renderer,scene,opts,_,_=old_load('frozen_static')
    state=torch.load(ROOT/p['parent'],map_location='cpu',weights_only=False)
    assert state['iteration']==2000 and state['stage']=='J6_I10_J4_RGB_only'
    scene.load_state_dict(state['model'],strict=True)
    for opt,value in zip(opts,state['optimizers']):opt.load_state_dict(value)
    assert r.bindings(scene,opts)==state['bindings'];restore_graphs(scene,state['physics_graphs'])
    scene.static.requires_grad_(False);scene.physics.requires_grad_(False)
    evidence=read(ROOT/p['evidence']);assert evidence['summary']['evidence_gate_passed']
    OBS=observations(evidence);TIMES=[f['time'] for f in read(DATA/'transforms_train.json')['frames']]
    restore_rng(state)
    r.assert_equivalent(scene.state_dict(),state['model'],0,0)
    for opt,value in zip(opts,state['optimizers']):r.assert_equivalent(opt.state_dict(),value,0,0)
    return renderer,scene,opts,None,state


def objective(renderer,scene,teacher,cam,t,target,enabled=False):
    assert not enabled,'Static depth trust is disabled in both T2 arms'
    rgb,package,_,_=rgb_objective(renderer,scene,None,cam,t,target,False)
    index=min(range(len(TIMES)),key=lambda i:abs(TIMES[i]-t));assert abs(TIMES[index]-t)<1e-9
    loss,track=add_tracks(rgb,scene,cam,t,OBS.get(index,[]),ARM=='rgb_tracks',read(BASE/'protocol.json')['track_weight'])
    return loss,package,rgb,track


def snapshot(scene,opts,step,stack,trace):
    state=SNAPSHOT(scene,opts,step,stack,trace)
    state.update(stage='T2_I10_mask_free_3D_tracks',evidence_sha256=sha(ROOT/read(BASE/'protocol.json')['evidence']),
        color_static_reference=False,old_I10_target_neutralization=False,static_fully_frozen=True)
    return state


def atomic(path,value):
    if isinstance(value,dict) and 'self_depth_loss' in value:
        value=dict(value);value['weighted_track_loss']=value.pop('self_depth_loss')
    ATOMIC(path,value)


def install(enabled=False):
    if not enabled:return r
    r.BASE=BASE;r.REPORT=REPORT;r.RUNS=RUNS;r.OUT=OUT
    r.load=load;r.objective=objective;r.snapshot=snapshot;r.atomic_json=atomic
    return r
