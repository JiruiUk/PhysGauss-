"""In-process adapter to frozen J4 training; all artifact paths belong to J6."""
from copy import deepcopy
import torch
from ..prepare_data import ROOT,read,sha
from ..rubber_joint import runtime as r
from ..rubber_joint.core import configure
from ..joint_rgb.core import restore_rng
from ..state_recovery.runtime import graph_state
from .core import merge_model

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_joint'
REPORT=BASE/'v1'
RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_joint/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_I10_joint/v1'
ORIGINAL_SNAPSHOT=r.snapshot


def load(arm):
    from ..motion_integrity.prepare import load_parent
    from ..union.runtime import restore
    from ..train_static import optimizer_for
    p=read(BASE/'protocol.json')
    for key in ['parent','static_parent']:assert sha(ROOT/p[key])==p[key+'_sha256']
    renderer,scene,dynamic_opt,_,_=load_parent()
    i10=restore(ROOT/p['parent'],scene,dynamic_opt,sha(ROOT/'experiments/staged_reconstruction/background_residual/protocol.json'))
    assert i10['iteration']==2000 and i10['phase']=='I10_static_residual'
    j4=torch.load(ROOT/p['static_parent'],map_location='cpu',weights_only=False)
    assert j4['iteration']==2000 and j4['protocol_sha256']==sha(ROOT/'experiments/staged_reconstruction/RubberDuck_joint_pilot/protocol.json')
    model=merge_model(i10['model'],j4['model'],enabled=True)
    scene.load_state_dict(model,strict=True)
    r.assert_equivalent(dynamic_opt.state_dict(),i10['optimizer'],0,0)
    static_opt=optimizer_for(scene.static);static_opt.load_state_dict(j4['optimizers'][0])
    r.assert_equivalent(static_opt.state_dict(),j4['optimizers'][0],0,0)
    for key,value in model.items():
        assert torch.equal(value,scene.state_dict()[key].detach().cpu()),key
        if key.startswith('static.') and key not in ['static._opacity','static._features_dc','static._features_rest']:
            assert torch.equal(i10['model'][key],j4['model'][key]),key
    r.assert_equivalent(graph_state(scene),i10['physics_graphs'],0,0)
    teacher=configure(scene,arm=='joint_depth_trust') if arm=='joint_depth_trust' else None
    scene.physics.requires_grad_(False)
    restore_rng(i10)
    parent=dict(i10);parent['model']=deepcopy(r.cpu_values(scene.state_dict()))
    return renderer,scene,[static_opt,dynamic_opt],teacher,parent


def snapshot(scene,opts,step,stack,trace):
    values=ORIGINAL_SNAPSHOT(scene,opts,step,stack,trace)
    p=read(BASE/'protocol.json')
    values.update(stage='J6_I10_J4_RGB_only',
        component_parents={'non_static':p['parent_sha256'],'static_and_static_Adam':p['static_parent_sha256']},
        old_I10_target_neutralization=False)
    return values


def install(enabled=False):
    if not enabled:
        return r
    r.BASE=BASE;r.REPORT=REPORT;r.RUNS=RUNS;r.OUT=OUT
    r.load=load;r.snapshot=snapshot
    return r
