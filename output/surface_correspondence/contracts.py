from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.surface_correspondence.common import *


def main():
    torch.set_num_threads(4);g,d,proxy=setup();metadata=read(TRAIN/'transforms_train.json')
    from reproduction.scripts.training_checkpoint import load_state
    full=load_state(DIRECTORY/'checkpoints/resume.pt')
    assert all(torch.allclose(v.cpu(),getattr(g,k).detach().cpu(),atol=1e-7,rtol=1e-6) for k,v in full['gaussian_parameters'].items())
    assert all(torch.equal(v.cpu(),d.grid4d.state_dict()[k].cpu()) for k,v in full['grid4d'].items())
    assert all(torch.equal(v.cpu(),d.deform.state_dict()[k].cpu()) for k,v in full['deform'].items());del full
    before=[x.clone() for x in [g.get_xyz,g.get_scaling,g.get_rotation,g.get_opacity]]
    image=np.load(ROOT/'output/surface_support/image_tracks.npz');rows=[]
    from variants.unified_flow.flow import contribution_weights
    for number in [2,86,200]:
        i=int(np.where(image['frames']==number)[0][0]);frame=next(f for f in metadata['frames'] if int(Path(f['file_path']).stem)==number)
        cam,warp,centers,means,cov,depth=geometry(g,d,frame,metadata)
        mass,alpha,relative=full_particle_mass(proxy,cam,warp,image['components'][i])
        color=torch.stack([(torch.arange(len(centers),device='cuda')%3==k).float() for k in range(3)],1)
        with torch.no_grad():package=render(proxy,cam,warp,color)
        assert torch.allclose(centers,package['means3D_deform'],atol=2e-6,rtol=2e-6)
        valid=package['radii']>0
        center_error=float((means[valid]-package['proj_means_2D'][valid]).abs().max());assert center_error<.003
        direct=[];adjoint=[]
        for k in range(4):
            mask=torch.as_tensor(image['components'][i]==k,device='cuda')
            direct.append(package['render'][:,mask].sum(1))
            adjoint.append(torch.stack([mass[torch.arange(len(centers),device='cuda')%3==j,k].sum() for j in range(3)]))
        direct=torch.stack(direct);adjoint=torch.stack(adjoint)
        binary_error=float(((direct-adjoint).abs()/direct.abs().clamp_min(1)).max());assert binary_error<5e-4
        safe,weights=contribution_weights(package,proxy.get_opacity)
        first20=float(weights.sum());full_image=float(alpha.sum())
        assert full_image>first20
        rows.append({'frame':number,'maximum_RGB_adjoint_relative_error':max(relative),'binary_cohort_integral_relative_error':binary_error,
             'projected_center_max_error_pixels':center_error,'full_image_alpha_mass':full_image,'first20_metadata_mass':first20,
             'recovered_mass_beyond_first20':float(mass[:,4].sum())-first20})
        print(rows[-1],flush=True)
    assert all(torch.equal(a,b) for a,b in zip(before,[g.get_xyz,g.get_scaling,g.get_rotation,g.get_opacity]))
    assert all(p.grad is None for p in [g._xyz,g._scaling,g._rotation,g._opacity])
    import diff_gaussian_rasterization,gaussian_renderer
    paths=[Path(__file__),ROOT/'output/surface_correspondence/common.py',Path(gaussian_renderer.__file__),Path(diff_gaussian_rasterization._C.__file__),
        DIRECTORY/'checkpoints/resume.pt',ROOT/'third_party/PIDG/submodules/flow-based-diff-gaussian-rasterization/cuda_rasterizer/backward.cu']
    write(REPORT/'renderer_contracts.json',{'passed':True,'C1_snapshot_matches_full_checkpoint':True,'zero_Gaussian_optimizer_updates':True,
        'no_model_values_or_gradients_changed':True,'full_contributions_beyond_first20_verified':True,'frames':rows,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}})


if __name__=='__main__':main()
