"""Declared analytic fixtures only; never imported by the dataset reconstructor."""
import numpy as np
import torch
from ..adaptive_static.geometry import patch_features
from .core import fit_slopes,plane_support,warped_cost


@torch.no_grad()
def measure_controls(protocol,device='cuda'):
    k=torch.tensor([[35.,0,31.5],[0,35.,31.5],[0,0,1.]],device=device)
    yy,xx=torch.meshgrid(torch.arange(64,device=device),torch.arange(64,device=device),indexing='ij')
    rx,ry=(xx-31.5)/35,(yy-31.5)/35
    uv=torch.tensor([[32.,30.],[29.,35.]],device=device); results={}
    for name,slope,shift in [('A_fractional',0.,.2),('B_integer',0.,8/35),('C_slanted',.4,.2)]:
        views=torch.eye(4,device=device).repeat(7,1,1)
        views[:,0,3]=torch.tensor([0,shift,2*shift,3*shift,-shift,-2*shift,-3*shift],device=device)
        images=[]; refdepth=None
        for j,v in enumerate(views):
            ox=-v[0,3]; z=(4-slope*ox)/(1+slope*rx)
            wx,wy=ox+rx*z,ry*z; g=.45+.15*torch.sin(wx*7)+.1*torch.cos(wy*9+wx*3)
            images.append(torch.stack([g,.9*g,.8*g]))
            if j==0: refdepth=z
        images=torch.stack(images); features,texture=patch_features(images,5)
        derived=fit_slopes(refdepth.cpu().numpy(),uv.long().cpu().numpy(),protocol)
        assert derived['slope_admitted'].all()
        slopes=torch.tensor(derived['coefficients'][:,:2],dtype=torch.float32,device=device)
        truth=refdepth[uv[:,1].long(),uv[:,0].long()]
        grid=1/torch.linspace(1/2,1/6,192,device=device)
        refs=features[0,:,uv[:,1].long(),uv[:,0].long()]; refvalid=texture[0,uv[:,1].long(),uv[:,0].long()]
        points,valid=plane_support(uv,grid[:,None].expand(-1,2),slopes,views[0],k,2,6)
        costs=torch.stack([warped_cost(points,valid,refs,refvalid,images[j],views[j],k) for j in range(1,7)])
        fit=costs[:4].sort(0).values[:3].mean(0).masked_fill(~valid,float('inf'))
        pt,pv=plane_support(uv,truth[None],slopes,views[0],k,2,6)
        truecost=torch.stack([warped_cost(pt,pv,refs,refvalid,images[j],views[j],k) for j in range(1,7)])[:,0]
        results[name]={'uv':uv.cpu().tolist(),'true_depth':truth.cpu().tolist(),
            'data_derived_slopes':derived['coefficients'][:,:2].tolist(),'minimum_depth':grid[fit.argmin(0)].cpu().tolist(),
            'true_fit':truecost[:4].sort(0).values[:3].mean(0).cpu().tolist(),
            'grid_depth':grid.cpu().tolist(),'fit_curve':[[float(v) if np.isfinite(v) else None for v in row] for row in fit.T.cpu().numpy()],
            'analytic_input_depths_only_for_control_not_dataset':True}
    return results
