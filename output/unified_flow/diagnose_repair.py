"""Read-only train-frame image/flow/momentum gradients and FD convergence."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
sys.path.insert(0,str(ROOT/'variants/unified_flow/code'))
torch.set_num_threads(4)
from variants.unified_flow.runtime import Tuning,CACHE
from variants.flow_comparison import temporal
from variants.unified_flow.run import assert_frozen
from scene_PIDG.gaussian_model import GaussianModel
spec=importlib.util.spec_from_file_location('repair_motion',ROOT/'output/flow_comparison/evaluate_motion_v2.py')
motion=importlib.util.module_from_spec(spec); spec.loader.exec_module(motion)
from utils.camera_utils import loadCam
import scene_PIDG.dataset_readers as loader
assert_frozen(ROOT/'variants/unified_flow/code')
OUT=ROOT/'experiments/momentum_repair'; OUT.mkdir(exist_ok=True)
source=ROOT/'data/derived/kinematic_resplit/training'
meta=motion.read(source/'transforms_train.json'); frames={int(Path(f['file_path']).stem):f for f in meta['frames']}
tree=ast.parse((ROOT/'variants/unified_flow/code/scene_PIDG/dataset_readers.py').read_text(encoding='utf-8'))
outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='readCamerasFromTransforms')
build=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='build_cam_info')
namespace=dict(vars(loader),path=str(source),extension='.png',white_background=False)
exec(compile(ast.fix_missing_locations(ast.Module(body=[build],type_ignores=[])),'actual_loader','exec'),namespace)
def camera(fid):
    frame=dict(frames[fid],time=(fid-1)/199)
    info=namespace['build_cam_info'](frame,meta['camera_angle_x'],False)
    return loadCam(SimpleNamespace(resolution=1600,data_device='cuda',load2gpu_on_the_fly=False),fid,info,1.)

rows=[]
for name in ['L0_corrected_trajectory','L1_shared_stress']:
    run=ROOT/'runs/pidg_unified_flow'/name/'rubber_duck'
    os.environ['PIDG_METRIC_CONFIG']=str(run/'metric_config.json')
    g,d=motion.load(run,'best'); tune=Tuning(run/'metric_config.json',OUT/name); tune.run.mkdir(exist_ok=True)
    manifest=motion.read(CACHE/'manifest.json'); tune.flow_pairs={p['source']:p for p in manifest['pairs']};tune.flow_tensors={}
    groups={'grid':list(d.grid4d.parameters()),'deformation':[p for k,p in d.deform.named_parameters() if not k.startswith('shared_stress.')],
        'stress':list((d.deform.shared_stress if name.startswith('L1') else g.velocity_net.stress_head).parameters())}
    groups={k:[p for p in v if p.requires_grad] for k,v in groups.items()}; params=[p for v in groups.values() for p in v]
    for frame in [2,83,155]:
        cam=camera(frame);pair=tune.flow_pairs[frame];tune.train_cameras={pair['target']:camera(pair['target'])}
        t=(frame-1)/199;warp=temporal.warp_at(d,g.get_xyz.detach(),t)
        image=motion.render(cam,g,motion.PIPE,motion.BACKGROUND,warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],False)['render']
        image_loss=tune.image_loss(50000,image,cam.original_image,SimpleNamespace(lambda_dssim=.2))
        flow=.01*tune.motion_loss(50000,g,d,cam,warp,motion.render,motion.PIPE,motion.BACKGROUND)
        momentum=.01*tune.momentum_loss(50000,g,d,cam.fid)
        gradients=[torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True) for loss in [image_loss,flow,momentum]]
        row={'model':name,'training_frame':frame,'image_loss':float(image_loss.detach()),'weighted_flow_loss':float(flow.detach()),'weighted_momentum_loss':float(momentum.detach()),'groups':{}}
        offset=0
        for group,ps in groups.items():
            gs=[gg[offset:offset+len(ps)] for gg in gradients];offset+=len(ps)
            norms=[sum(float(x.detach().double().square().sum()) for x in gg if x is not None)**.5 for gg in gs]
            dot=sum(float((a.detach().double()*b.detach().double()).sum()) for a,b in zip(gs[0],gs[2]) if a is not None and b is not None)
            row['groups'][group]={'image_norm':norms[0],'flow_norm':norms[1],'momentum_norm':norms[2],
                'momentum_to_image':norms[2]/norms[0] if norms[0]>1e-20 else None,
                'momentum_image_cosine':dot/(norms[0]*norms[2]) if norms[0]*norms[2]>1e-20 else None}
        with torch.no_grad():
            x=g.get_xyz.detach()[torch.arange(128,device='cuda')*len(g.get_xyz)//128]
            f=lambda t:temporal.position(d,x,t)
            # Finest reference is a numerical comparison, not exact physical acceleration.
            ref=temporal.derivative(f,t,1/199/16,2); vals=[]
            for h in [1/199*2,1/199,1/199/2,1/199/4,1/199/8,1/199/16]:
                a=temporal.derivative(f,t,h,2)
                vals.append({'h':h,'acceleration_rms':float(a.square().mean().sqrt()),'relative_difference_to_finest':float((a-ref).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-8))})
            row['FD_step_sweep']=vals
        rows.append(row);print(json.dumps(row),flush=True)
        del gradients,gs,image_loss,flow,momentum,image,warp
    del g,d,tune,params,groups
    torch.cuda.empty_cache()
assert_frozen(ROOT/'variants/unified_flow/code')
(OUT/'diagnosis.json').write_text(json.dumps({'status':'complete','probes':rows,'protocol':'Three fixed training frames, actual loader RGB/cameras at1600, deterministic deformation; same-batch image/flow/momentum. No optimizer updates or checkpoint writes. Image probe excludes TV/coverage terms.','limitations':'Post-training local probes; not all historical training batches or a causal ablation. Finest finite difference is not physical GT.'},indent=2)+'\n')
