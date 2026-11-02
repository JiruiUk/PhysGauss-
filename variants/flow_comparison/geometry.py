"""Pixel-coordinate camera compensation and confidence; independent of PIDG imports."""
import torch
import torch.nn.functional as F


def pixel_grid(h,w,like):
    y,x=torch.meshgrid(torch.arange(h,device=like.device,dtype=like.dtype),torch.arange(w,device=like.device,dtype=like.dtype),indexing='ij')
    return torch.stack([x,y],-1)


def sample(field,coords):
    h,w=field.shape[:2];grid=coords.clone()
    grid[...,0]=2*grid[...,0]/max(w-1,1)-1;grid[...,1]=2*grid[...,1]/max(h-1,1)-1
    scalar=field.ndim==2
    if scalar:field=field[...,None]
    result=F.grid_sample(field.permute(2,0,1)[None],grid[None],mode='bilinear',padding_mode='zeros',align_corners=True)[0].permute(1,2,0)
    return result[...,0] if scalar else result


def project(points,w2c,K):
    cam=points@w2c[:3,:3].T+w2c[:3,3]
    pixel=cam@K.T
    return pixel[...,:2]/pixel[...,2:].clamp_min(1e-8),cam[...,2]


def motion_target(forward,depth_next,w2c_current,K_current,w2c_next,K_next):
    h,w=depth_next.shape;grid=pixel_grid(h,w,forward);next_uv=grid+forward
    d=sample(depth_next,next_uv)
    rays=torch.cat([next_uv,torch.ones_like(next_uv[...,:1])],-1)@torch.linalg.inv(K_next).T
    cam=rays*d[...,None];c2w=torch.linalg.inv(w2c_next)
    world=cam@c2w[:3,:3].T+c2w[:3,3]
    returned,z=project(world,w2c_current,K_current)
    inside=(next_uv[...,0]>=0)&(next_uv[...,0]<=w-1)&(next_uv[...,1]>=0)&(next_uv[...,1]<=h-1)
    valid=inside&(d>1e-6)&(z>1e-6)&torch.isfinite(returned).all(-1)
    return returned-grid,valid


def forward_backward_valid(forward,backward):
    h,w=forward.shape[:2];dest=pixel_grid(h,w,forward)+forward;back=sample(backward,dest)
    inside=(dest[...,0]>=0)&(dest[...,0]<=w-1)&(dest[...,1]>=0)&(dest[...,1]<=h-1)
    threshold=.01*(forward.square().sum(-1)+back.square().sum(-1))+.5
    return inside&((forward+back).square().sum(-1)<=threshold)


def flow_loss(pred,target,valid):
    if pred.shape!=target.shape or pred.shape[-1]!=2:raise ValueError('Expected H,W,2 flows')
    h,w=pred.shape[:2];scale=pred.new_tensor([w,h])
    difference=torch.where(valid[...,None]>0,pred-target,torch.zeros_like(pred))
    error=(difference/scale).abs().mean(-1)
    return (error*valid).sum()/valid.sum().clamp_min(1)
