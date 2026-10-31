"""Frozen final-test identity visualizations after the selection lock only."""
import argparse
import json
import os
import runpy
import sys
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from .run import ROOT,HERE,RUNS,REPORTS,NAMES,engine,base,assert_frozen
from variants.object_centers.motion_review import palette
from variants.object_centers.audit import tensor_image
from .model import rotation


def export(smoke=False):
    if not smoke and not (RUNS/'selection_lock.json').exists():raise ValueError('Test identity exports still locked')
    os.environ['PIDG_HASH_SOURCE_DIR']=str(engine.SOURCE/'hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR']=str(engine.SOURCE/'tmp_build')
    repo=HERE/'code';assert_frozen(repo);sys.path.insert(0,str(repo))
    from arguments import ModelParams,PipelineParams
    from scene_PIDG import Scene,DeformModel
    from gaussian_renderer import GaussianModel,render
    from variants.kinematic_resplit.runtime import configure_scene_from_env
    root=RUNS/'_smoke' if smoke else RUNS
    output=root/'identity_exports';output.mkdir(exist_ok=True);summaries=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
    for name in NAMES[1:]:
        directory=root/name/'rubber_duck';os.environ['PIDG_METRIC_CONFIG']=str(directory/'metric_config.json')
        for tag in (('100',) if smoke else ('best','50000')):
            parser=argparse.ArgumentParser();model=ModelParams(parser);pipe=PipelineParams(parser);args=parser.parse_args([])
            for k,v in runpy.run_path(str(directory/'model_config.py')).items():
                if not k.startswith('__'):setattr(args,k,v)
            args.source_path=str(base.DATA/('training' if smoke else 'final_test'));args.model_path=str(directory)
            args.eval=True;args.resolution=800;args.load2gpu_on_the_fly=True
            folder=output/name/tag;folder.mkdir(parents=True,exist_ok=True)
            with torch.no_grad():
                gaussians=GaussianModel(args.sh_degree);scene=Scene(model.extract(args),gaussians,load_iteration=tag,shuffle=False)
                configure_scene_from_env(scene,args.source_path)
                deform=DeformModel(args.grid_args,args.network_args,scale_xyz=args.scale_xyz,reg_spatial_able=False,reg_temporal_able=False)
                deform.load_weights(str(directory),iteration=tag)
                module=deform.object_model
                module.initialize(gaussians.get_xyz.detach())
                p=module.assignments(gaussians.get_xyz.detach());confidence,labels=p.max(-1)
                rgb=torch.from_numpy(palette(module.capacity)).to(p)[labels]
                rgb[confidence<.5]=rgb.new_tensor([.35,.4,.46])
                np.savez_compressed(folder/'kernel_identity.npz',labels=labels.cpu().numpy(),confidence=confidence.cpu().numpy(),
                    active=module.active.cpu().numpy(),canonical=gaussians.get_xyz.cpu().numpy())
                interventions=[]
                views=scene.getTestCameras()
                for index,view in enumerate(views):
                    view.load2device();t=view.fid.unsqueeze(0).expand(len(p),-1)
                    pkg=deform.step(gaussians.get_xyz.detach(),t,fixed_attention=True)
                    raw=render(view,gaussians,pipe.extract(args),torch.zeros(3,device='cuda'),pkg['d_xyz'],pkg['d_rotation'],pkg['d_scaling'])
                    colored=render(view,gaussians,pipe.extract(args),torch.zeros(3,device='cuda'),pkg['d_xyz'],pkg['d_rotation'],pkg['d_scaling'],override_color=rgb)
                    panels=[tensor_image(view.original_image[:3]),tensor_image(raw['render']),tensor_image(colored['render'])]
                    canvas=Image.new('RGB',(2400,500),'#edf2f7');d=ImageDraw.Draw(canvas)
                    for col,(title,image) in enumerate(zip([f'GT / frame{view.image_name}',name+' / '+tag,'Learned/frozen grouping; gray: pmax<0.5'],panels)):
                        d.text((col*800+10,12),title,font=font,fill='#17304a');canvas.paste(image,(col*800,50))
                    canvas.save(folder/f'{index:05d}.png')
                    if index in (0,12,25,37,49):
                        # Permute identity weights only, leaving motion slots fixed.
                        from .runtime import blend
                        x=gaussians.get_xyz.detach();normal=module.grouped(x,t,p)
                        shuffled=module.grouped(x,t,p.roll(1,dims=1))
                        changed=pkg['d_xyz'].clone();changed[:,4:]+=blend(deform)*(shuffled-normal)
                        alt=render(view,gaussians,pipe.extract(args),torch.zeros(3,device='cuda'),changed,pkg['d_rotation'],pkg['d_scaling'])
                        interventions.append({'frame':view.image_name,'RGB_L1_change_after_identity_shuffle':float((alt['render']-raw['render']).abs().mean()),
                            'meaning':'branch-use diagnostic, not semantic ID accuracy'})
                    view.load2device('cpu')
                record={'model':name,'snapshot':tag,'source_iteration':deform.object_iteration,'images':len(views),
                    'split':'validation' if smoke else 'test',
                    'capacity':module.capacity,'active_candidates':int(module.active.sum()),'mean_max_assignment':float(confidence.mean()),
                    'confident_kernel_fraction':float((confidence>=.5).float().mean()),'shuffled_identity_diagnostics':interventions,
                    'ground_truth_instance_accuracy_available':False,'gallery':str(folder/'index.html')}
                engine.write_json(folder/'summary.json',record);summaries.append(record)
                page='''<!doctype html><meta charset="utf-8"><title>Object ID diagnostic</title><style>body{background:#142033;color:#edf5ff;font:16px system-ui;margin:24px}img{width:100%}a{color:#9be0b8}</style><h1>GT | render | object identity</h1><p>Gray: assignment confidence below0.5. Colors are not semantic truth or material IDs.</p><input id="f" type="range" min="0" max="49" value="0"><a id="a">Open PNG</a><img id="i"><script>function u(){let p=String(+f.value).padStart(5,'0')+'.png';i.src=p;a.href=p}f.oninput=u;u();</script>'''
                engine.write_text_atomic(folder/'index.html',page.replace('max="49"',f'max="{len(views)-1}"'))
    engine.write_json(REPORTS/('identity_preflight.json' if smoke else 'identity_exports.json'),{'status':'completed','summaries':summaries})
    engine.write_text_atomic(output/'index.html','<!doctype html><meta charset="utf-8"><h1>Object identity diagnostics</h1>'+''.join(
        f'<p><a href="{s["model"]}/{s["snapshot"]}/index.html">{s["model"]} / {s["snapshot"]}</a></p>' for s in summaries))
