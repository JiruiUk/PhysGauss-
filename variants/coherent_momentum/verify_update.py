"""Real smoke checkpoint -> isolated CUDA update; never write model weights."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
sys.path.insert(0,str(ROOT/'variants/coherent_momentum/code'))
from .runtime import Tuning, Parent
from .run import NAME, RUNS, REPORTS
from variants.metric_tuning.checkpoint import base
from variants.unified_flow.runtime import CACHE
from variants.unified_flow.camera_contract import loader_fovs
from variants.flow_comparison.temporal import warp_at
from scene_PIDG.gaussian_model import GaussianModel
from scene_PIDG.deform_model import DeformModel
from scene_PIDG.cameras import Camera
from gaussian_renderer import render


def main():
    run=RUNS/'_smoke'/NAME/'rubber_duck'
    os.environ['PIDG_METRIC_CONFIG']=str(run/'metric_config.json')
    state=base.load_state(run/'checkpoints/resume.pt');opt=SimpleNamespace(**state['opt']);ds=state['dataset']
    g=GaussianModel(3);d=DeformModel(ds['grid_args'],ds['network_args'],reg_spatial_able=False,reg_temporal_able=False)
    d.train_setting(opt);base.restore_models(state,g,d,opt)
    tune=Tuning(run/'metric_config.json',REPORTS/'optimizer_probe');tune.run.mkdir(exist_ok=True);tune.config['verify_capture_raw']=True
    manifest=json.loads((CACHE/'manifest.json').read_text());tune.flow_pairs={p['source']:p for p in manifest['pairs']};tune.flow_tensors={}
    meta=json.loads((ROOT/'data/derived/kinematic_resplit/training/transforms_train.json').read_text())
    def camera(fid):
        f=next(f for f in meta['frames'] if int(Path(f['file_path']).stem)==fid)
        matrix=np.linalg.inv(np.asarray(f['transform_matrix']));r=-matrix[:3,:3].T;r[:,0]=-r[:,0]
        fx,fy=loader_fovs(meta['camera_angle_x'])
        return Camera(fid,r,-matrix[:3,3],fx,fy,torch.zeros(3,225,400),None,f'{fid:04d}',fid,fid=(fid-1)/199)
    cam=camera(82);tune.train_cameras={tune.flow_pairs[82]['target']:camera(tune.flow_pairs[82]['target'])}
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False);bg=torch.zeros(3,device='cuda')
    warp=warp_at(d,g.get_xyz.detach(),81/199)
    image=render(cam,g,pipe,bg,warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],False)['render']
    image_loss=tune.image_loss(100,image,cam.original_image,opt)
    guarded=tune.momentum_loss(100,g,d,cam.fid)
    # Same deterministic graph/sample and unmodified parameters; private gradient parity.
    raw=tune._verification_raw
    private=list(d.deform.shared_stress.parameters())+[g.force_sigma]
    a=torch.autograd.grad(raw,private,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(guarded,private,retain_graph=True,allow_unused=True)
    assert all(x is None and y is None or x is not None and y is not None and torch.equal(x,y) for x,y in zip(a,b))
    assert torch.equal(raw.detach(),guarded.detach())
    record=json.loads((tune.run/'momentum_gradient_guard.jsonl').read_text().splitlines()[-1])
    groups={'grid':list(d.grid4d.parameters()),'deformation':[p for n,p in d.deform.named_parameters() if not n.startswith('shared_stress.')]}
    for key,ps in groups.items():
        grad=torch.autograd.grad(opt.physics_loss_weight*guarded,ps,retain_graph=True,allow_unused=True)
        norm=sum(float(v.detach().double().square().sum()) for v in grad if v is not None)**.5
        assert norm<=.1*record['groups'][key]['image_norm']*(1+2e-4)
    flow=tune.motion_loss(100,g,d,cam,warp,render,pipe,bg)
    modules={'grid':d.grid4d,'deformation':d.deform,'shared_stress':d.deform.shared_stress,'legacy':g.velocity_net}
    selected={k:next(m.parameters()) for k,m in modules.items()}
    before={k:p.detach().clone() for k,p in selected.items()}
    g.optimizer.zero_grad(set_to_none=True);d.optimizer.zero_grad(set_to_none=True)
    loss=image_loss+opt.physics_loss_weight*guarded+.01*flow;loss.backward()
    assert all(p.grad is None or p.grad.isfinite().all() for m in modules.values() for p in m.parameters())
    assert not any(p.grad is not None for p in g.velocity_net.parameters())
    g.optimizer.step();d.optimizer.step()
    changed={k:float((p.detach()-before[k]).abs().sum()) for k,p in selected.items()}
    assert all(changed[k]>0 for k in ['grid','deformation','shared_stress']) and changed['legacy']==0
    assert all(p.isfinite().all() for m in modules.values() for p in m.parameters())
    result={'status':'passed','actual_parameter_update_l1':changed,'private_gradients_exact':True,
        'real_shared_gradient_caps_verified':True,'true_forward_equal':True,'source_checkpoint_untouched':True,'guard':record}
    (REPORTS/'optimizer_update.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
