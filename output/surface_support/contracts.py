"""Independent selected-frame renderer contracts and input availability audit."""
from pathlib import Path
import sys,json,hashlib,os
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.surface_support.audit import MODELS,TRAIN,REPORT,read,write,sha
from output.surface_support.core import RenderProxy


@torch.no_grad()
def main():
    from variants.c1_flow_ablation.run import assert_frozen
    assert_frozen(ROOT/'variants/c1_flow_ablation/code')
    from variants.c1_flow_ablation.results import motion_module
    m=motion_module();directory,_,_=MODELS['C1_final'];g,d=m.load(directory,'50000')
    from variants.shape_coherence.observations import camera,PIPE
    from variants.flow_comparison.temporal import warp_at
    from variants.kinematic_tuning.model import center_trajectory
    from variants.flow_comparison.flow import projected,symmetric
    from gaussian_renderer import render
    import gaussian_renderer,diff_gaussian_rasterization
    from reproduction.scripts.training_checkpoint import load_state
    full=load_state(directory/'checkpoints/resume.pt')
    assert all(torch.allclose(v.cpu(),getattr(g,k).detach().cpu(),atol=1e-7,rtol=1e-6) for k,v in full['gaussian_parameters'].items())
    assert all(torch.equal(v.cpu(),d.grid4d.state_dict()[k].cpu()) for k,v in full['grid4d'].items())
    assert all(torch.equal(v.cpu(),d.deform.state_dict()[k].cpu()) for k,v in full['deform'].items());del full
    os.environ['PIDG_METRIC_CONFIG']=str(directory/'metric_config.json')
    meta=read(TRAIN/'transforms_train.json');background=torch.zeros(3,device='cuda')
    first=warp_at(d,g.get_xyz.detach(),1/199);refscale=(g.get_scaling+first['d_scaling']).clone();refrot=(g.get_rotation+first['d_rotation']).clone()
    fixed=RenderProxy(g,get_scaling=refscale,get_rotation=refrot)
    before=[g.get_xyz.clone(),g.get_scaling.clone(),g.get_rotation.clone(),g.get_opacity.clone()];checks=[]
    for number in [2,74,86,200]:
        frame=next(f for f in meta['frames'] if int(Path(f['file_path']).stem)==number)
        cam=camera(frame,meta,400);w=warp_at(d,g.get_xyz.detach(),(number-1)/199)
        c=center_trajectory(g.get_xyz.detach(),w['d_xyz']);color=torch.ones(len(c),3,device='cuda')
        p=render(cam,g,PIPE,background,w['d_xyz'],w['d_rotation'],w['d_scaling'],override_color=color)
        assert torch.allclose(c,p['means3D_deform'],atol=2e-6,rtol=2e-6)
        mean,cov=projected(c,g.get_scaling+w['d_scaling'],g.get_rotation+w['d_rotation'],cam)
        valid=p['radii']>0
        pe=float((mean[valid]-p['proj_means_2D'][valid]).abs().max())
        ce=float((torch.linalg.inv(cov[valid])-symmetric(p['conic_2D'])[valid]).abs().max())
        ae=float((p['render'][0]-p['alpha'].squeeze()).abs().max())
        assert pe<.003 and ce<.003 and ae<1e-6
        actual_centers=p['means3D_deform'].clone();del p
        q=render(cam,fixed,PIPE,background,w['d_xyz'],torch.zeros_like(w['d_rotation']),torch.zeros_like(w['d_scaling']),override_color=color)
        assert torch.equal(actual_centers,q['means3D_deform']);del q
        checks.append({'frame':number,'projected_center_max_pixel_error':pe,'inverse_covariance_max_error':ce,'white_render_vs_full_alpha_max_error':ae,'fixed_footprint_centers_identical':True})
    assert all(torch.equal(a,b) for a,b in zip(before,[g.get_xyz,g.get_scaling,g.get_rotation,g.get_opacity]))
    masks=[]
    for f in meta['frames']:
        number=int(Path(f['file_path']).stem);p=ROOT/f'data/PIDG/rubber_duck/mask/{number:04d}.png'
        with Image.open(p) as im:a=np.asarray(im);values=np.unique(a).tolist()
        assert values==[0,255]
        masks.append({'frame':number,'values':values,'sha256':sha(p)})
    source=[Path(gaussian_renderer.__file__),Path(diff_gaussian_rasterization._C.__file__),
      ROOT/'third_party/PIDG/submodules/flow-based-diff-gaussian-rasterization/cuda_rasterizer/forward.cu',
      ROOT/'third_party/PIDG/submodules/flow-based-diff-gaussian-rasterization/rasterize_points.cu',
      ROOT/'variants/flow_comparison/flow.py',ROOT/'variants/flow_comparison/geometry.py',ROOT/'variants/unified_flow/flow.py',
      Path(__file__),ROOT/'tests/test_surface_support_advection.py']
    write(REPORT/'renderer_contracts.json',{'passed':True,'frozen_C1_flow_evaluation_source_passed':True,'C1_snapshot_matches_full_checkpoint':True,'selected_train_frame_contracts':checks,
      'no_getter_or_parameter_mutation':True,'binary_mask_training_frames':masks,'mask_is_not_instance_ID':True,
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source}})
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
