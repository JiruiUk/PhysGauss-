"""Read-only full-compositor compatibility check of candidate image tracks.

Does not assign Gaussian IDs or modify reconstruction parameters. A bad score
can reflect representation support failure OR an uncertain image association.
"""
from pathlib import Path
import sys,os,json
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.motion_correspondence.run import REPORT,OUT,read,sha,write
PALETTE=np.array([[235,90,100],[40,210,180],[245,170,65],[175,130,245]])/255


def labelled(base,labels):
    result=np.repeat(base.mean(-1,keepdims=True),3,-1)*.6
    for k in range(4):result[labels==k]=.3*base[labels==k]+.7*PALETTE[k]
    return result


@torch.no_grad()
def main():
    from variants.c1_flow_ablation.results import motion_module
    from variants.c1_flow_ablation.run import assert_frozen
    assert_frozen(ROOT/'variants/c1_flow_ablation/code');m=motion_module()
    from gaussian_renderer import render
    from variants.shape_coherence.observations import camera,PIPE
    from variants.flow_comparison.temporal import warp_at
    from variants.kinematic_tuning.model import center_trajectory
    from variants.flow_comparison.flow import projected
    directory=ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck'
    checkpoint=directory/'checkpoints/resume.pt';before=sha(checkpoint)
    assert before=='b5d4cf5c9fb8fba4fcc26124555d48aa7d3a963a8be1d28488954752db486480'
    g,d=m.load(directory,'50000');os.environ['PIDG_METRIC_CONFIG']=str(directory/'metric_config.json')
    gp=ROOT/'output/shape_coherence/audit/reference_graph.pt';graph=torch.load(gp,map_location='cuda',weights_only=True)
    groups=graph['labels'];known=groups>=0;n=len(groups);color=torch.zeros(n,3,device='cuda');fourth=torch.zeros_like(color)
    for k in range(3):color[:,k]=(groups==k)
    fourth[:,0]=(groups==3)
    meta=read(m.TRAIN/'transforms_train.json');frames={int(Path(f['file_path']).stem):f for f in meta['frames']}
    old=np.load(ROOT/'output/surface_support/image_tracks.npz')
    cv=np.load(OUT/'history.npz');guard=np.load(OUT/'history_guarded.npz')
    guarded={r['frame']:r for r in read(REPORT/'history_guarded.json')['frames']}
    (OUT/'render_support').mkdir(exist_ok=True);rows=[]
    for number in [2,42,43,55,63,74,86,108,134,172,200]:
        i=int(np.where(old['frames']==number)[0][0]);frame=frames[number];cam=camera(frame,meta,400)
        w=warp_at(d,g.get_xyz.detach(),(number-1)/199)
        a=render(cam,g,PIPE,m.BACKGROUND,w['d_xyz'],w['d_rotation'],w['d_scaling'],override_color=color)
        alpha=a['alpha'].squeeze().clone();channels=a['render'].clone();del a
        b=render(cam,g,PIPE,m.BACKGROUND,w['d_xyz'],w['d_rotation'],w['d_scaling'],override_color=fourth)
        mass=torch.cat([channels,b['render'][:1]],0);del b
        lab=guard['labels'][i];duck=torch.from_numpy(old['components'][i]>=0).cuda();den=alpha[duck].sum()
        per=[]
        centers=center_trajectory(g.get_xyz.detach(),w['d_xyz'])
        mean,_=projected(centers[known],(g.get_scaling+w['d_scaling'])[known],(g.get_rotation+w['d_rotation'])[known],cam)
        kl=groups[known].cpu().numpy();means=mean.cpu().numpy()
        for k in range(4):
            mask=lab==k;mk=torch.from_numpy(mask).cuda();total=alpha[mk].sum()
            points=means[kl==k];xy=np.rint(points).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<400)&(xy[:,1]>=0)&(xy[:,1]<225)
            on=inside.copy();on[inside]=mask[xy[inside,1],xy[inside,0]]
            per.append({'seed_ID':k,'accepted_image_pixels':int(mask.sum()),
               'early_patch_full_alpha_share_on_candidate':float(mass[k][mk].sum()/total) if float(total)>1e-8 else None,
               'early_patch_centers_on_candidate_fraction':float(on.mean()) if mask.any() else None})
        totalshare=float(mass.sum(0)[duck].sum()/den)
        rows.append({'frame':number,'all_early_patch_full_alpha_share_on_yellow':totalshare,'per_candidate':per,
                     'not_physical_identity_accuracy':True})
        with Image.open(m.TRAIN/(frame['file_path']+'.png')) as im:rgba=np.asarray(im.convert('RGBA').resize((400,225),Image.Resampling.BILINEAR))/255
        gt=rgba[...,:3]*rgba[...,3:4]
        colors=mass.permute(1,2,0).cpu().numpy()@PALETTE
        overlay=Image.fromarray((gt*255).round().astype('uint8'));draw=ImageDraw.Draw(overlay)
        for uv,k in zip(means,kl):
            x,y=uv
            if np.isfinite(uv).all() and 0<=x<400 and 0<=y<225:draw.ellipse((x-.6,y-.6,x+.6,y+.6),fill=tuple((PALETTE[k]*255).astype(int)))
        values=[gt,labelled(gt,old['labels'][i]),labelled(gt,cv['labels'][i]),labelled(gt,lab),colors,np.asarray(overlay)/255]
        titles=[f'train GT {number}','old FB visible evidence','history-only candidates','guarded candidates / unknown','C1 full early-patch contribution','C1 early centers on GT']
        canvas=Image.new('RGB',(2400,265),'#182a3d');draw=ImageDraw.Draw(canvas)
        for j,(value,title) in enumerate(zip(values,titles)):
            canvas.paste(Image.fromarray((np.clip(value,0,1)*255).round().astype('uint8')),(400*j,40));draw.text((400*j+6,10),title,fill='white')
        canvas.save(OUT/'render_support'/f'frame{number:04d}.png')
        print('render compatibility',number,totalshare,flush=True)
    assert sha(checkpoint)==before
    source={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),gp,checkpoint]}
    write(REPORT/'render_support.json',{'read_only':True,'new_Gaussian_optimizer_steps':0,'full_compositor_not_first20':True,
      'models_unchanged':True,'source_sha256':source,'frames':rows,
      'interpretation':'Candidate image identity is not material ID. Matching image tracks does not repair missing canonical Gaussian surface support.'})


if __name__=='__main__':main()
