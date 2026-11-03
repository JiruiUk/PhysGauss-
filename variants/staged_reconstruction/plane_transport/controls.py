"""Fixed analytic transport fixtures; not imported by RGB dataset inference."""
import numpy as np
import torch
from ..adaptive_static.geometry import patch_features
from ..rgb_plane_warp.core import fit_slopes,plane_support
from .core import propose


@torch.no_grad()
def measure_controls(protocol,device='cuda'):
    from ..prepare_data import ROOT,read
    slope_protocol=read(ROOT/'experiments/staged_reconstruction/S1_rgb_plane_warp/protocol.json')
    k=torch.tensor([[35.,0,31.5],[0,35.,31.5],[0,0,1.]],device=device)
    yy,xx=torch.meshgrid(torch.arange(64,device=device),torch.arange(64,device=device),indexing='ij')
    rx,ry=(xx-31.5)/35,(yy-31.5)/35
    uv=torch.tensor([[32.,30.],[29.,35.]],device=device)
    all_uv=torch.stack([xx.flatten(),yy.flatten()],1).float(); results={}
    for name,slope,shift in [('A_fractional',0.,.2),('B_integer',0.,8/35),('C_slanted',.4,.2)]:
        views=torch.eye(4,device=device).repeat(7,1,1)
        views[:,0,3]=torch.tensor([0,shift,2*shift,3*shift,-shift,-2*shift,-3*shift],device=device)
        images=[]; depths=[]; slopes=[]; admission=[]
        for j,v in enumerate(views):
            ox=-v[0,3]; z=(4-slope*ox)/(1+slope*rx)
            wx,wy=ox+rx*z,ry*z; g=.45+.15*torch.sin(wx*7)+.1*torch.cos(wy*9+wx*3)
            images.append(torch.stack([g,.9*g,.8*g])); depths.append(z)
            data=fit_slopes(z.cpu().numpy(),all_uv.long().cpu().numpy(),slope_protocol)
            s=torch.tensor(data['coefficients'][:,:2],dtype=torch.float32,device=device)
            _,valid=plane_support(all_uv,z.flatten()[None],s,v,k,2,6)
            admitted=torch.tensor(data['slope_admitted'],device=device)&valid[0]
            slopes.append(s.reshape(64,64,2)); admission.append(admitted.reshape(64,64))
        images=torch.stack(images); features,texture=patch_features(images,5)
        maps={'original':torch.stack(depths),'slopes':torch.stack(slopes),'admitted':torch.stack(admission)}
        original=torch.full((2,),4.8,device=device)
        old_slopes=maps['slopes'][0,uv[:,1].long(),uv[:,0].long()]
        updated,d=propose(uv,original,old_slopes,maps,images,features,texture,views,k,0,list(range(1,7)),2,6,True)
        results[name]={'uv':uv.cpu().tolist(),'true_depth':maps['original'][0,uv[:,1].long(),uv[:,0].long()].cpu().tolist(),
            'original_depth':original.cpu().tolist(),'updated_depth':updated.cpu().tolist(),'accepted':d['accepted'].cpu().tolist(),
            'fit_old':d['fit_old'].cpu().tolist(),'fit_new':d['fit_new'].cpu().tolist(),
            'source_uv':d['source_uv'].cpu().tolist(),'candidate_valid':d['candidate_valid'].cpu().tolist(),
            'candidate_depth':d['raw_candidate_depth'].cpu().tolist(),'selected_source':d['selected_source'].cpu().tolist(),
            'analytic_input_planes_only_for_control_never_dataset':True}
    return results
