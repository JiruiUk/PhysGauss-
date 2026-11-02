"""Known early-frame image components lifted conservatively to candidate kernel IDs."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from .model import clear_duck_masks,fixed_graph

EARLY=[2,6,10,14,19]
PIPE=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False)


def camera(frame, metadata, width=800):
    from scene_PIDG.cameras import Camera
    from variants.unified_flow.camera_contract import loader_fovs
    matrix=np.linalg.inv(np.array(frame['transform_matrix']))
    rotation=-matrix[:3,:3].T;rotation[:,0]=-rotation[:,0]
    fovx,fovy=loader_fovs(metadata['camera_angle_x'])
    number=int(Path(frame['file_path']).stem)
    return Camera(number,rotation,-matrix[:3,3],fovx,fovy,torch.zeros(3,width*9//16,width),None,
                  f'{number:04d}',number,fid=(number-1)/199)


@torch.no_grad()
def infer_reference(g,d,metadata,data,output=None,width=800):
    from gaussian_renderer import render
    from variants.flow_comparison.temporal import warp_at
    from variants.kinematic_tuning.model import center_trajectory
    from variants.unified_flow.flow import contribution_weights
    frames={int(Path(f['file_path']).stem):f for f in metadata['frames']}
    if not set(EARLY)<=set(frames):raise ValueError('Clear reference frames must be train-only')
    n=len(g.get_xyz);mass=torch.zeros(n,5,device='cuda');votes=torch.zeros(n,4,device='cuda')
    reference=None;panels=[];palette=np.array([[235,90,100],[40,210,180],[245,170,65],[175,130,245]],dtype=np.uint8)
    for number in EARLY:
        frame=frames[number];cam=camera(frame,metadata,width)
        with Image.open(data/(frame['file_path']+'.png')) as image:
            rgba=np.array(image.convert('RGBA').resize((width,width*9//16),Image.Resampling.BILINEAR))
        masks=clear_duck_masks(rgba)
        pkg=warp_at(d,g.get_xyz.detach(),(number-1)/199)
        if reference is None:reference=center_trajectory(g.get_xyz.detach(),pkg['d_xyz']).detach().clone()
        render_pkg=render(cam,g,PIPE,torch.zeros(3,device='cuda'),pkg['d_xyz'],pkg['d_rotation'],pkg['d_scaling'])
        safe,weights=contribution_weights(render_pkg,g.get_opacity)
        foreground=torch.from_numpy((rgba[...,3]>.95*255)).cuda()
        target=torch.from_numpy(masks).cuda();target=torch.where(target>=0,target,torch.full_like(target,4))
        per=torch.zeros_like(mass)
        # RGB is a mixture: assigning only the largest contributor excludes most
        # surface kernels. Lift every meaningful PRE-update alpha*T contribution.
        normalized=weights/weights.sum(0,keepdim=True).clamp_min(1e-8)
        for layer in range(len(weights)):
            valid=foreground&(normalized[layer]>=.05)&(weights[layer]>=1/255)
            per.view(-1).scatter_add_(0,(safe[layer][valid]*5+target[valid]).reshape(-1),weights[layer][valid])
        mass+=per;votes+=(per[:,:4]>=.02)
        if output:
            rgb=rgba[:,:,:3].copy();rgb[masks>=0]=(rgb[masks>=0]*.35+palette[masks[masks>=0]]*.65).astype('uint8')
            panels.append(rgb)
    winning,label=mass[:,:4].max(-1)
    confidence=winning/mass.sum(-1).clamp_min(1e-8)
    count=votes.gather(1,label[:,None])[:,0]
    supported=(confidence>=.95)&(winning>=.2)&(count>=3)&(g.get_opacity[:,0]>=.03)
    labels=torch.where(supported,label,torch.full_like(label,-1))
    counts=[int((labels==i).sum()) for i in range(4)]
    if min(counts)<30:raise ValueError(f'Insufficient reliable lift for four ducks: {counts}')
    a,neighbors,valid=fixed_graph(reference,labels)
    graph={'reference':reference,'labels':labels,'anchors':a,'neighbors':neighbors,'valid':valid,
           'point_count':n,'frame_ids':EARLY,'confidence':confidence}
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.concatenate(panels,axis=0)).save(output/'early_training_components.png')
    return graph
