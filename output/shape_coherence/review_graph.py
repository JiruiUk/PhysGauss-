"""Read-only 16k candidate gate evidence. Does not approve the gate automatically."""
from pathlib import Path
import sys
import json
import os
from types import SimpleNamespace
import torch
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.shape_coherence.run import RUNS,REPORTS,OUT,NAME,assert_frozen
from variants.shape_coherence.observations import camera,PIPE
from variants.metric_tuning import run as engine


@torch.no_grad()
def main():
    assert_frozen(ROOT/'variants/shape_coherence/code')
    path=RUNS/NAME/'rubber_duck';checkpoint=path/'checkpoints/resume.pt'
    receipt=engine.read_json(checkpoint.with_suffix('.json'))
    assert receipt['iteration']==16000 and receipt['sha256']==engine.sha256(checkpoint)
    from variants.c1_flow_ablation.results import motion_module
    m=motion_module()
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.deform_model import DeformModel
    from reproduction.scripts.training_checkpoint import load_state,restore_models
    import runpy
    state=load_state(checkpoint);args=runpy.run_path(str(path/'model_config.py'))
    g=GaussianModel(3);d=DeformModel(args['grid_args'],args['network_args'],reg_spatial_able=False,reg_temporal_able=False)
    opt=SimpleNamespace(**state['opt']);d.train_setting(opt);restore_models(state,g,d,opt)
    os.environ['PIDG_METRIC_CONFIG']=str(path/'metric_config.json')
    graph=torch.load(path/'shape_reference/reference_graph.pt',map_location='cuda',weights_only=True)
    side=torch.load(checkpoint.with_suffix('.shape.pt'),map_location='cpu',weights_only=True)
    assert side['iteration']==16000 and side['checkpoint_sha256']==receipt['sha256']
    assert engine.read_json(checkpoint.with_suffix('.metrics.json'))['shape_graph_sha256']==engine.sha256(checkpoint.with_suffix('.shape.pt'))
    assert all(torch.equal(v.cpu(),side['graph'][k]) for k,v in graph.items() if torch.is_tensor(v))
    from gaussian_renderer import render
    from variants.flow_comparison.temporal import warp_at
    from variants.kinematic_tuning.model import center_trajectory
    from variants.color_repair.color import colors as actual_colors
    from variants.unified_flow.flow import contribution_weights
    metadata=engine.read_json(m.TRAIN/'transforms_train.json')
    frames={int(Path(f['file_path']).stem):f for f in metadata['frames']}
    output=OUT/'candidate_graph';output.mkdir(parents=True,exist_ok=True)
    palette=torch.tensor([[.92,.35,.39],[.16,.82,.71],[.96,.67,.25],[.69,.51,.96]],device='cuda')
    known=graph['labels']>=0;rows=[]
    for number in [2,19,50,62,74,86,98,108,146,200]:
        frame=frames[number];cam=camera(frame,metadata,800)
        pkg=warp_at(d,g.get_xyz.detach(),(number-1)/199)
        centers=center_trajectory(g.get_xyz.detach(),pkg['d_xyz'])
        rgb=render(cam,g,PIPE,m.BACKGROUND,pkg['d_xyz'],pkg['d_rotation'],pkg['d_scaling'])
        colors=actual_colors(g,centers,cam).clamp(0,1).mean(-1,keepdim=True).expand(-1,3).clone()*.35
        colors[known]=palette[graph['labels'][known]]
        lifted=render(cam,g,PIPE,m.BACKGROUND,pkg['d_xyz'],pkg['d_rotation'],pkg['d_scaling'],override_color=colors)
        with Image.open(m.TRAIN/(frame['file_path']+'.png')) as image:
            rgba=np.asarray(image.convert('RGBA').resize((800,450),Image.Resampling.BILINEAR))/255
        gt=rgba[...,:3]*rgba[...,3:4]
        duck=(rgba[...,3]>.95)&(gt[...,0]>.25)&(gt[...,1]>.18)&(gt[...,2]<.8*np.minimum(gt[...,0],gt[...,1]))
        safe,weights=contribution_weights(rgb,g.get_opacity)
        mass=(weights*known[safe]).sum(0);mask=torch.from_numpy(duck).cuda()
        fraction=float(mass[mask].sum()/weights.sum(0)[mask].sum().clamp_min(1e-8))
        rows.append({'frame':number,'known_contribution_on_duck_pixels':fraction})
        canvas=Image.new('RGB',(2400,490),'#182a3d');draw=ImageDraw.Draw(canvas)
        for i,(value,title) in enumerate([(gt,'train GT'),(rgb['render'].permute(1,2,0).clamp(0,1).cpu().numpy(),'candidate16k'),(lifted['render'].permute(1,2,0).clamp(0,1).cpu().numpy(),'reference patches; unknown dimmed')]):
            canvas.paste(Image.fromarray(np.round(value*255).astype('uint8')),(800*i,40));draw.text((800*i+8,10),f'{title} | frame {number}',fill='white')
        canvas.save(output/f'frame{number:04d}.png')
    report={'stage':'16k before shape updates','approved':False,'manual_review_pending':True,
        'checkpoint_sha256':receipt['sha256'],'graph_sha256':engine.sha256(path/'shape_reference/reference_graph.pt'),
        'kernel_counts':[int((graph['labels']==i).sum()) for i in range(4)],'anchors':len(graph['anchors']),
        'reference_labels_are_partial_and_not_physical_GT':True,'per_frame':rows}
    engine.write_json(REPORTS/'candidate_graph_evidence.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
