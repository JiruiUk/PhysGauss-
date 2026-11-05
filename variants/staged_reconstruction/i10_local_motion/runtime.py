"""Artifact-isolated full-state local experiment; frozen PIDG remains intact."""
import numpy as np
import torch
from ..i10_tracking.io import ROOT,DATA,read,sha
from ..rubber_joint import runtime as r
from ..rubber_joint.core import objective as rgb_objective
from ..i10_joint.runtime import load as old_load
from ..joint_rgb.core import restore_rng
from ..state_recovery.runtime import restore_graphs
from ..i10_track_train.core import add_tracks
from .core import support,add_local

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_local_motion'
REPORT=BASE/'v1'
RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_local_motion/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_I10_local_motion/v1'
SNAPSHOT=r.snapshot
ATOMIC=r.atomic_json
ARM=None
OBS=None
PAIRS=None
TIMES=None
CONFIG=None
LAST={}


def load(arm):
    global ARM,OBS,PAIRS,TIMES,CONFIG
    assert arm in ['audited_tracks','local_motion']
    ARM=arm;CONFIG=read(BASE/'protocol.json');assert sha(ROOT/CONFIG['parent'])==CONFIG['parent_sha256']
    renderer,scene,opts,_,_=old_load('frozen_static')
    state=torch.load(ROOT/CONFIG['parent'],map_location='cpu',weights_only=False)
    assert state['stage']=='J6_I10_J4_RGB_only' and state['iteration']==2000
    scene.load_state_dict(state['model'],strict=True)
    for opt,value in zip(opts,state['optimizers']):opt.load_state_dict(value)
    assert r.bindings(scene,opts)==state['bindings'];restore_graphs(scene,state['physics_graphs'])
    scene.static.requires_grad_(False);scene.physics.requires_grad_(False)
    with np.load(ROOT/CONFIG['trajectories']) as f:
        OBS,PAIRS,_=support(read(ROOT/CONFIG['evidence']),f['xyz'])
    TIMES=[f['time'] for f in read(DATA/'transforms_train.json')['frames']]
    restore_rng(state)
    r.assert_equivalent(scene.state_dict(),state['model'],0,0)
    for opt,value in zip(opts,state['optimizers']):r.assert_equivalent(opt.state_dict(),value,0,0)
    return renderer,scene,opts,None,state


def objective(renderer,scene,teacher,cam,t,target,enabled=False):
    assert not enabled,'Static trust is off in both local-study arms'
    rgb,package,_,_=rgb_objective(renderer,scene,None,cam,t,target,False)
    index=min(range(len(TIMES)),key=lambda i:abs(TIMES[i]-t));assert abs(TIMES[index]-t)<1e-9
    tracked,track=add_tracks(rgb,scene,cam,t,OBS.get(index,[]),True,CONFIG['track_weight'])
    local=rgb*0;loss=tracked
    if ARM=='local_motion' and index in PAIRS:
        loss,local=add_local(tracked,scene.dynamic_at(t).get_xyz,PAIRS[index],True,CONFIG['local_weight'],CONFIG['local_scale'])
    LAST.update(weighted_audited_track_loss=float(track.detach()),weighted_local_loss=float(local.detach()),
                supported_tracks=len(OBS.get(index,[])),supported_pairs=len(PAIRS.get(index,[])))
    return loss,package,rgb,track+local


def snapshot(scene,opts,step,stack,trace):
    state=SNAPSHOT(scene,opts,step,stack,trace)
    state.update(stage='L1_I10_supported_local_motion',evidence_sha256=sha(ROOT/read(BASE/'protocol.json')['evidence']),
                 color_static_reference=False,old_I10_target_neutralization=False,static_fully_frozen=True,
                 original_global_coverage_not_promoted=True,local_prior_is_parent_conditioned=True)
    return state


def atomic(path,value):
    if isinstance(value,dict) and 'self_depth_loss' in value:
        value=dict(value);value['weighted_auxiliary_loss']=value.pop('self_depth_loss');value.update(LAST)
    ATOMIC(path,value)


def install(enabled=False):
    if not enabled:return r
    r.BASE=BASE;r.REPORT=REPORT;r.RUNS=RUNS;r.OUT=OUT
    r.load=load;r.objective=objective;r.snapshot=snapshot;r.atomic_json=atomic
    return r
