"""Read-only C1 shape audit. Only training images and immutable C1 weights."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from variants.shape_coherence.observations import infer_reference,camera,PIPE
from variants.shape_coherence.model import local_shape


def main():
    from variants.c1_flow_ablation.results import motion_module
    m=motion_module()
    source=ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck'
    os.environ['PIDG_METRIC_CONFIG']=str(source/'metric_config.json')
    from variants.flow_comparison.temporal import warp_at
    from variants.kinematic_tuning.model import center_trajectory
    from gaussian_renderer import render
    out=ROOT/'output/shape_coherence/audit';out.mkdir(parents=True,exist_ok=True)
    reports=ROOT/'experiments/shape_coherence';reports.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((m.TRAIN/'transforms_train.json').read_text())
    frames={int(Path(f['file_path']).stem):f for f in metadata['frames']}
    g,d=m.load(source,'50000')
    with torch.no_grad():
        graph=infer_reference(g,d,metadata,m.TRAIN,out)
        torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in graph.items()},out/'reference_graph.pt')
        rows=[];chosen=sorted(set(min(frames,key=lambda n:abs(n-t)) for t in [2,19,34,50,62,70,74,78,82,86,90,94,98,102,108,118,146,182,200]))
        from variants.color_repair.color import colors as actual_colors
        from variants.unified_flow.flow import contribution_weights
        palette=torch.tensor([[.92,.35,.39],[.16,.82,.71],[.96,.67,.25],[.69,.51,.96]],device='cuda')
        known=graph['labels']>=0;reference_scale=None
        for number in chosen:
            pkg=warp_at(d,g.get_xyz.detach(),(number-1)/199)
            centers=center_trajectory(g.get_xyz.detach(),pkg['d_xyz'])
            stats=local_shape(graph['reference'],centers,graph['anchors'],graph['neighbors'],graph['valid'])
            cam=camera(frames[number],metadata,800)
            rgb=render(cam,g,PIPE,m.BACKGROUND,pkg['d_xyz'],pkg['d_rotation'],pkg['d_scaling'])
            base_colors=actual_colors(g,centers,cam).clamp(0,1)
            colors=base_colors.mean(-1,keepdim=True).expand(-1,3).clone()*.35
            colors[known]=palette[graph['labels'][known]]
            identity=render(cam,g,PIPE,m.BACKGROUND,pkg['d_xyz'],pkg['d_rotation'],pkg['d_scaling'],override_color=colors)
            with Image.open(m.TRAIN/(frames[number]['file_path']+'.png')) as im:
                rgba=np.array(im.convert('RGBA').resize((800,450),Image.Resampling.BILINEAR)).astype('float32')/255
            gt=rgba[:,:,:3]*rgba[:,:,3:4]
            predicted=rgb['render'].permute(1,2,0).clamp(0,1).cpu().numpy()
            fg=rgba[:,:,3]>.95;duck=(gt[:,:,0]>.25)&(gt[:,:,1]>.18)&(gt[:,:,2]<.8*np.minimum(gt[:,:,0],gt[:,:,1]))&fg
            error=np.mean(np.abs(gt-predicted),axis=2)
            scale=g.get_scaling+pkg['d_scaling']
            selected=scale[known]
            if reference_scale is None:reference_scale=selected.abs().clone()
            ratio=selected.abs()/reference_scale.clamp_min(1e-6)
            safe,weights=contribution_weights(rgb,g.get_opacity)
            known_mass=(weights*known[safe]).sum(0)
            duck_tensor=torch.from_numpy(duck).cuda()
            # Rendered scale signs cancel in covariance; do not call negative scale an inversion.
            row={'frame':number,'duck_pixel_MAE':float(error[duck].mean()),'foreground_MAE':float(error[fg].mean()),
                 'affine_residual':float(stats['affine']),'collapse_penalty':float(stats['collapse']),
                 'sv_min_p05':float(stats['singular'][:,-1].quantile(.05)),
                 'sv_max_p95':float(stats['singular'][:,0].quantile(.95)),
                 'nonpositive_det_fraction':float((stats['det']<=0).float().mean()),
                 'scale_sign_negative_fraction_not_physical_inversion':float((selected<=0).float().mean()),
                 'scale_abs_ratio_to_clear_reference_p95':float(ratio.quantile(.95)),
                 'known_kernel_opacity_mass_on_duck_pixels':float(known_mass[duck_tensor].sum()/weights.sum(0)[duck_tensor].sum().clamp_min(1e-8)),
                 'opacity_mean':float(g.get_opacity[known].mean())}
            rows.append(row)
            panel=Image.new('RGB',(2400,490),'#182a3d');draw=ImageDraw.Draw(panel)
            for i,(image,title) in enumerate([(gt,'GT training'),(predicted,'C1 render'),(identity['render'].permute(1,2,0).clamp(0,1).cpu().numpy(),'reference patches; unknown dimmed actual color')]):
                panel.paste(Image.fromarray(np.round(image*255).astype('uint8')),(800*i,40));draw.text((800*i+10,10),f'{title} | frame {number}',fill='white')
            panel.save(out/f'frame{number:04d}.png')
            print(json.dumps(row),flush=True)
        summary={'source':'historical C1 final50000, immutable','train_images_only':True,'reference_frames':graph['frame_ids'],
                 'kernel_counts':[int((graph['labels']==i).sum()) for i in range(4)],'unknown_kernels':int((graph['labels']<0).sum()),
                 'neighborhoods':len(graph['anchors']),'reference_graph_sha256':m.sha(out/'reference_graph.pt'),
                 'opacity_is_time_invariant':True,'candidate_IDs_are_not_physical_GT':True,'per_frame':rows,
                 'source_snapshot_sha256':{str(p.relative_to(ROOT)):m.sha(p) for p in [source/'point_cloud/iteration_50000/point_cloud.ply',source/'deform/iteration_50000/deform.pth']}}
        (reports/'diagnosis.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
