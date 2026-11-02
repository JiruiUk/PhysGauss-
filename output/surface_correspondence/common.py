from pathlib import Path
import sys,json,hashlib,os
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
REPORT=ROOT/'experiments/surface_correspondence';OUT=ROOT/'output/surface_correspondence'
CACHE=ROOT/'runs/pidg_surface_correspondence/cache'
TRAIN=ROOT/'data/derived/kinematic_resplit/training'
FLOW=ROOT/'data/derived/flow_comparison_train_only'
DIRECTORY=ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def setup():
    from variants.c1_flow_ablation.run import assert_frozen
    from variants.c1_flow_ablation.results import motion_module
    assert_frozen(ROOT/'variants/c1_flow_ablation/code');m=motion_module()
    assert sha(DIRECTORY/'checkpoints/resume.pt')=='b5d4cf5c9fb8fba4fcc26124555d48aa7d3a963a8be1d28488954752db486480'
    g,d=m.load(DIRECTORY,'50000');os.environ['PIDG_METRIC_CONFIG']=str(DIRECTORY/'metric_config.json')
    from output.surface_support.core import RenderProxy
    proxy=RenderProxy(g,get_xyz=g.get_xyz.detach(),get_scaling=g.get_scaling.detach(),get_rotation=g.get_rotation.detach(),get_opacity=g.get_opacity.detach())
    return g,d,proxy


def geometry(g,d,frame,metadata):
    from variants.shape_coherence.observations import camera
    from variants.flow_comparison.temporal import warp_at
    from variants.kinematic_tuning.model import center_trajectory
    from variants.flow_comparison.flow import projected
    number=int(Path(frame['file_path']).stem);cam=camera(frame,metadata,400)
    with torch.no_grad():
        warp=warp_at(d,g.get_xyz.detach(),(number-1)/199)
        centers=center_trajectory(g.get_xyz.detach(),warp['d_xyz'])
        means,covariance=projected(centers,g.get_scaling+warp['d_scaling'],g.get_rotation+warp['d_rotation'],cam)
        view=cam.world_view_transform;depth=(centers@view[:3,:3]+view[3,:3])[:,2]
    return cam,warp,centers,means,covariance,depth


def render(g,cam,warp,color):
    from gaussian_renderer import render as rr
    from variants.shape_coherence.observations import PIPE
    return rr(cam,g,PIPE,torch.zeros(3,device='cuda'),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=color)


def full_particle_mass(proxy,cam,warp,components):
    """Full RGB color adjoint, with numerical checks, beyond layer20."""
    target=torch.as_tensor(components,device='cuda')
    masks=torch.stack([*(target==k for k in range(4)),torch.ones_like(target,dtype=torch.bool)]).float()
    values=[];relative=[];alpha=None
    for start in [0,3]:
        batch=torch.zeros(3,*target.shape,device='cuda');count=min(3,5-start);batch[:count]=masks[start:start+count]
        color=torch.ones(len(proxy.get_xyz),3,device='cuda',requires_grad=True)
        package=render(proxy,cam,warp,color)
        image=package['render'];a=package['alpha'].squeeze().detach()
        assert float((image.detach()-a[None]).abs().max())<1e-6
        gradient=torch.autograd.grad((image*batch).sum(),color)[0].detach()
        expected=(a[None]*batch).sum((1,2));actual=gradient.sum(0)
        error=(actual-expected).abs()/expected.clamp_min(1)
        assert float(error.max())<5e-4,(actual.tolist(),expected.tolist())
        assert torch.isfinite(gradient).all() and float(gradient.min())>=-1e-7
        values.append(gradient[:,:count]);relative.append(float(error.max()));alpha=a
        del package,image,color,gradient
    return torch.cat(values,1),alpha,relative
