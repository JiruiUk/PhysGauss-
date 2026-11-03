"""Read-only static/dynamic compositing and loss-gradient diagnostic."""
from pathlib import Path
import math
import torch
import numpy as np
from PIL import Image, ImageDraw

from variants.staged_reconstruction.prepare_data import ROOT, DATA, REPORT, read, write, sha


def pixels(value):
    return (value.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)


def main():
    torch.set_num_threads(4)
    from variants.staged_reconstruction.early_depth.train import load_initial
    from variants.staged_reconstruction.train_static import camera, PIPE
    from variants.staged_reconstruction.stage2.model import compose
    from variants.staged_reconstruction.stage2.train import image
    from variants.staged_reconstruction.union.train import reconstruction
    from variants.staged_reconstruction.union.core import dynamic_mask
    from variants.staged_reconstruction.covariance_transport.runtime import install, renderer_for, with_covariance
    from variants.staged_reconstruction.covariance_transport.core import covariance, pack
    from variants.staged_reconstruction.covariance_transport.freeze import verify
    verify()
    output=REPORT/'static_dynamic_coupling'
    assert not (output/'audit.json').exists(), 'Preserve audit versions'
    output.mkdir(parents=True,exist_ok=True)
    out=ROOT/'output/staged_reconstruction/static_dynamic_coupling'
    out.mkdir(parents=True,exist_ok=True)
    paths={
        'S3U_final6k':(ROOT/'runs/pidg_staged_reconstruction/S3_union/physics/formal/checkpoint.pt','8c767b7835eee936e49989369d7bc0277d7567d3aedaa7385c06654284b68ad0'),
        'I8_preview7000':(ROOT/'runs/pidg_staged_reconstruction/covariance_transport/previews/step07000/checkpoint.pt','6a498afc8b694bfaab94ed05927d11dd5fe82279f7baa1ba1eb00bd3a21d3994')}
    meta=read(DATA/'transforms_train.json')
    byid={int(Path(frame['file_path']).stem):frame for frame in meta['frames']}
    results={}
    for arm,(path,digest) in paths.items():
        assert sha(path)==digest
        renderer,scene,opt,initial,_=load_initial()
        del opt,initial
        if arm.startswith('I8'):
            install(scene,read(REPORT/'covariance_transport/protocol.json'),True)
            renderer=renderer_for(renderer)
        state=torch.load(path,map_location='cpu',weights_only=False)
        scene.load_state_dict(state['model']); del state
        groups={'canonical_xyz':[scene.pools[0]._xyz],
                'deformation_fields':list(scene.grids.parameters())+list(scene.networks.parameters())}
        records=[]
        for number in [43,86,146]:
            frame=byid[number]; cam=camera(frame,meta,800)
            rgba=image(frame,800)
            gt=torch.tensor(rgba[...,:3]*rgba[...,3:4],device='cuda').permute(2,0,1)
            package=scene.render(renderer,cam,frame['time']); pred=package['render']
            r,g,b=gt
            stat=(b>r*1.3+.02)&(b>g*1.05)
            dyn=torch.tensor(dynamic_mask(rgba),device='cuda')
            static_loss=.5*(pred[:,stat]-gt[:,stat]).abs().mean()
            dynamic_loss=.5*(pred[:,dyn]-gt[:,dyn]).abs().mean()
            total=reconstruction(package,gt,rgba,scene,cam)
            gradients={}
            for name,parameters in groups.items():
                gs=torch.autograd.grad(static_loss,parameters,retain_graph=True,allow_unused=True)
                gd=torch.autograd.grad(dynamic_loss,parameters,retain_graph=True,allow_unused=True)
                ga=torch.autograd.grad(total,parameters,retain_graph=True,allow_unused=True)
                def norm(values):
                    return math.sqrt(sum(float(value.detach().square().sum()) for value in values if value is not None))
                ns,nd,na=norm(gs),norm(gd),norm(ga)
                dot=sum(float((s.detach()*d.detach()).sum()) for s,d in zip(gs,gd) if s is not None and d is not None)
                gradients[name]={'weighted_static_term_gradient_norm':ns,'weighted_dynamic_term_gradient_norm':nd,
                                 'complete_image_objective_gradient_norm':na,'static_vs_dynamic_gradient_cosine':dot/max(ns*nd,1e-30)}
            with torch.no_grad():
                frozen=renderer(cam,scene.static,PIPE,scene.centers.new_zeros(3),0.,0.,0.)['render']
                static_only_mae=(frozen[:,stat]-gt[:,stat]).abs().mean()
                combined_mae=(pred[:,stat]-gt[:,stat]).abs().mean()
            records.append({'observed_training_frame':number,'width':800,'gradient_groups':gradients,
                            'static_only_MAE_on_conditional_blue_mask':float(static_only_mae),
                            'combined_MAE_on_same_mask':float(combined_mae)})
            del package,pred,total,static_loss,dynamic_loss,gs,gd,ga
        probes=[]
        with torch.no_grad():
            for camid,number in [(188,43),(188,146),(86,86)]:
                cam=camera(byid[camid],meta,1600); time=(number-1)/199
                dynamic=scene.dynamic_at(time)
                alone=renderer(cam,dynamic,PIPE,scene.centers.new_zeros(3),0.,0.,0.)
                full=scene.render(renderer,cam,time)
                bank=compose(scene.static,[dynamic])
                if hasattr(dynamic,'transported_covariance'):
                    with_covariance(bank,torch.cat([pack(covariance(scene.static.get_scaling,scene.static.get_rotation)),dynamic.transported_covariance]))
                color=scene.centers.new_ones(len(bank.get_xyz),3)
                color[:len(scene.static.get_xyz)]=0
                visible=renderer(cam,bank,PIPE,scene.centers.new_zeros(3),0.,0.,0.,override_color=color)['render'][0]
                independent=alone['alpha'].squeeze()
                assert float((visible-independent).max())<3e-4
                mask=independent>.05
                row={'camera':camid,'frame':number,'dynamic_alpha_mass':float(independent.sum()),
                     'visible_dynamic_weight_with_static':float(visible.sum()),
                     'fraction_dynamic_weight_attenuated_by_static':float((independent-visible).sum()/independent.sum()),
                     'mean_dynamic_alpha_on_its_above005_support':float(independent[mask].mean())}
                probes.append(row)
                folder=out/arm; folder.mkdir(parents=True,exist_ok=True)
                im1=Image.fromarray(pixels(full['render'])); im2=Image.fromarray(pixels(alone['render']))
                im1.save(folder/f'c{camid}_t{number}_combined_native.png')
                im2.save(folder/f'c{camid}_t{number}_dynamic_only_native.png')
                sheet=Image.new('RGB',(1600,480),'#171a22'); draw=ImageDraw.Draw(sheet)
                for k,(label,im) in enumerate([('combined',im1),('dynamic only; same trained checkpoint',im2)]):
                    sheet.paste(im.resize((800,450),Image.Resampling.LANCZOS),(800*k,30))
                    draw.text((800*k+8,8),f'{arm} c{camid}/t{number}: {label}',fill='white')
                sheet.save(folder/f'c{camid}_t{number}_comparison.png')
                print('STATIC_DYNAMIC_COUPLING',arm,camid,number,row['fraction_dynamic_weight_attenuated_by_static'],flush=True)
        assert sha(path)==digest
        results[arm]={'checkpoint_sha256':digest,'observed_training_gradient_probes':records,'native_novel_view_probes':probes}
        del scene,renderer
        import gc
        gc.collect(); torch.cuda.empty_cache()
    write(output/'audit.json',{'source_sha256':sha(Path(__file__)),'results':results,'optimizer_updates':0,'test_RGB_read':False,
          'scope':'Three observed train-frame gradient probes and three native novel-view probes per frozen arm. Static occlusion may be correct; attenuation is not static extraction error. Static-term gradients also respond to dynamic leakage. Removing static at inference does not undo its initialization/training influence. These are not training causal ablations or final completeness metrics.',
          'image_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(out.rglob('*.png'))}})


if __name__=='__main__':main()
