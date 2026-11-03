"""Measure the frozen matcher without exporting privileged pixel arrays."""
import numpy as np
import torch
from ..adaptive_static.geometry import source_cost, project_torch, patch_features
from ..rgb_prior_proposals.core import accept_rgb
from ..synthetic_audit.generate import box, sphere


def scene_hit(origin, rays, time, static_only=False):
    # Evaluator geometry, exactly the unchanged synthetic generator's scene.
    depths = [box(origin, rays, [-.8,-.75,-.6], [.8,-.45,.6])[0],
              box(origin, rays, [-.2,-.45,-.2], [.2,.9,.2])[0],
              box(origin, rays, [.42,-.45,-.32], [.46,.6,.32])[0]]
    if not static_only:
        depths += [sphere(origin, rays, [.8*np.sin(2*np.pi*time),.28,.7*np.cos(2*np.pi*time)], .25)[0],
                   box(origin, rays, [-.45+.6*time,.62,-.7], [-.2+.6*time,.85,-.45])[0]]
    return np.min(depths, axis=0)


def world_points(uv, depth, view, k):
    inverse = np.linalg.inv(view)
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(k).T
    return (rays*depth[:, None]) @ inverse[:3,:3].T + inverse[:3,3]


def single_face(depth, normal, visible):
    """All 25 original pixels must see the same normal, including at borders."""
    h, w = depth.shape
    center = normal[2:h-2,2:w-2]
    inner = np.ones((h-4,w-4), bool)
    for dy in range(5):
        for dx in range(5):
            d = depth[dy:dy+h-4,dx:dx+w-4]
            n = normal[dy:dy+h-4,dx:dx+w-4]
            v = visible[dy:dy+h-4,dx:dx+w-4]
            inner &= v & np.isfinite(d) & (d > 0) & ((n*center).sum(-1) > .99)
    result = np.zeros((h,w), bool); result[2:h-2,2:w-2] = inner
    return result


def nearest_grid(depth, near, far):
    grid = 1 / torch.linspace(1/near, 1/far, 192, dtype=torch.float32, device=depth.device)
    index = torch.round((1/depth-1/near)/((1/far-1/near)/191)).long().clamp(0,191)
    return grid[index]


@torch.no_grad()
def measure(uv, depths, features=None, texture=None, views=None, k=None,
            index=None, sources=None, enabled=False):
    if not enabled:
        return depths
    assert depths.shape[0] == 3 and len(sources) == 6
    ref = features[index,:,uv[:,1].long(),uv[:,0].long()]
    rays = torch.cat([uv,torch.ones_like(uv[:,:1])],1) @ torch.linalg.inv(k).T
    inv = torch.linalg.inv(views[index])
    points = (rays[None]*depths[...,None]) @ inv[:3,:3].T + inv[:3,3]
    costs = torch.stack([source_cost(points,ref,features[j],texture[j],views[j],k) for j in sources],1)
    fit = costs[:,:4].sort(1).values[:,:3].mean(1)
    accepted = []
    for hyp in [1,2]:
        _, mask = accept_rgb(depths[0],depths[hyp],fit[0],fit[hyp],costs[0,4:],costs[hyp,4:],
                             torch.ones_like(depths[0],dtype=torch.bool),True)
        accepted.append(mask)
    return costs, fit, torch.stack(accepted)


@torch.no_grad()
def true_visibility(points, views_np, times, sources, views, k, height, width):
    result = []
    pts = torch.as_tensor(points,dtype=torch.float32,device=views.device)[None]
    for j in sources:
        origin = np.linalg.inv(views_np[j])[:3,3]
        hit = scene_hit(origin,points-origin,float(times[j]))
        margin = project_torch(pts,views[j],k,height,width)[1][0].cpu().numpy()
        result.append(margin & np.isfinite(hit) & (np.abs(hit-1) <= 1e-4))
    return np.stack(result)


def controls(device='cpu'):
    """Three preregistered scenes; saved curves are analytic controls only."""
    k = torch.tensor([[35.,0,31.5],[0,35.,31.5],[0,0,1.]],device=device)
    uv = torch.tensor([[32.,30.],[29.,35.]],device=device)
    yy,xx = torch.meshgrid(torch.arange(64,device=device),torch.arange(64,device=device),indexing='ij')
    rays = torch.stack([(xx-31.5)/35,(yy-31.5)/35,torch.ones_like(xx)],-1)
    output = {}
    for name,slope,shift in [('A_fractional',0.,.2),('B_integer',0.,8/35),('C_slanted',.4,.2)]:
        views = torch.eye(4,device=device).repeat(7,1,1)
        views[:,0,3] = torch.tensor([0,shift,2*shift,3*shift,-shift,-2*shift,-3*shift],device=device)
        images = []
        for v in views:
            ox = -v[0,3]
            z = (4-slope*ox)/(rays[...,2]+slope*rays[...,0])
            wx,wy = ox+rays[...,0]*z,rays[...,1]*z
            g = .45+.15*torch.sin(wx*7)+.1*torch.cos(wy*9+wx*3)
            images.append(torch.stack([g,.9*g,.8*g]))
        features,texture = patch_features(torch.stack(images),5)
        grid = 1/torch.linspace(1/2,1/6,192,device=device)
        ref = features[0,:,uv[:,1].long(),uv[:,0].long()]
        ray = torch.cat([uv,torch.ones_like(uv[:,:1])],1) @ torch.linalg.inv(k).T
        costs = torch.stack([source_cost(ray[None]*grid[:,None,None],ref,features[j],texture[j],views[j],k) for j in range(1,7)])
        fit = costs[:4].sort(0).values[:3].mean(0)
        gt = 4/(1+slope*ray[:,0]); grid_gt = nearest_grid(gt,2,6)
        cs,fs,_ = measure(uv,torch.stack([torch.full_like(gt,4.8),gt,grid_gt]),features,texture,views,k,0,list(range(1,7)),True)
        output[name] = {'uv':uv.cpu().tolist(),'true_depth':gt.cpu().tolist(),
            'grid_depth':grid.cpu().tolist(),'fit_curve':fit.T.cpu().tolist(),
            'minimum_depth':grid[fit.argmin(0)].cpu().tolist(),
            'true_fit':fs[1].cpu().tolist(),'nearest_grid_fit':fs[2].cpu().tolist(),
            'true_source_costs':cs[1].T.cpu().tolist()}
    return output
