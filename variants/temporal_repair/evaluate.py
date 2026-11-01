"""Validation-first, streaming CUDA ablation of train-knot temporal readouts."""
import argparse
from datetime import datetime,timezone
import html
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
REPO=ROOT/'variants/integrity_tuning/code';sys.path.insert(0,str(REPO))
sys.path.append(str(ROOT/'reproduction/scripts'))
os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from variants.temporal_repair.model import bracket,repaired_state
from variants.integrity_tuning.model import image_metrics,aggregate
from variants.coverage_tuning.data import read_alpha
from variants.metric_tuning.model import quantize_png
from variants.metric_tuning import run as engine
RUN=ROOT/'runs/pidg_integrity_tuning/H3_combined/rubber_duck'
REPORTS=ROOT/'experiments/temporal_repair'
RUNS=ROOT/'runs/pidg_temporal_repair'
METHODS={'native':(0.,False),'center_half':(.5,False),'center_full':(1.,False),'state_lerp_raw':(1.,True)}


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def save_tensor(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    a=(value.detach().cpu().clamp(0,1).numpy()*255).round().astype('uint8')
    if a.ndim==3:a=np.moveaxis(a,0,-1)
    if a.shape[-1:]==(1,):a=a[...,0]
    Image.fromarray(a).save(path)


def monitor(state):
    write(REPORTS/'readout_status.json',state)
    page='<meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Temporal repair</title><style>body{font:16px system-ui;margin:28px;background:#142033;color:#edf5ff}pre{white-space:pre-wrap}a{color:#8de4b1}</style><h1>H3 最差时刻诊断与时间轨迹修正</h1>'
    page+=f"<h2>{html.escape(state['status'])}</h2><p>{state.get('completed',0)} / {state.get('total',30)} frames | 每5秒刷新</p><pre>"+html.escape(json.dumps(state,ensure_ascii=False,indent=2))+'</pre>'
    page+='<p><a href="readout_validation.json">Validation metrics</a></p>'
    (REPORTS/'monitor.html').write_text(page,encoding='utf-8')


def main():
    own=argparse.ArgumentParser();own.add_argument('--split',choices=['validation','test'],default='validation');args=own.parse_args()
    REPORTS.mkdir(parents=True,exist_ok=True);RUNS.mkdir(parents=True,exist_ok=True)
    selection=None
    if args.split=='test':
        selection=engine.read_json(REPORTS/'readout_selection.json')
        if not selection.get('locked_before_test'):raise ValueError('Validation selection must be locked before test')
    source=ROOT/'data/derived/kinematic_resplit'/('training' if args.split=='validation' else 'final_test')
    output=RUNS/'readout_ablation'/args.split
    from arguments import ModelParams,PipelineParams,merge_config
    from scene_PIDG import Scene,GaussianModel,DeformModel
    from gaussian_renderer import render
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    from pytorch_msssim import ms_ssim
    import lpips
    parser=argparse.ArgumentParser();mp=ModelParams(parser);pp=PipelineParams(parser)
    parsed=parser.parse_args(['-s',str(source),'-m',str(RUN),'--resolution','1600','--load2gpu_on_the_fly'])
    parsed=merge_config(parsed,str(RUN/'model_config.py'));parsed.data_device='cuda:0'
    os.environ['PIDG_METRIC_CONFIG']=str(RUN/'metric_config.json')
    os.environ['PIDG_HASH_SOURCE_DIR']=str(engine.SOURCE/'hashencoder');os.environ['PIDG_HASH_BUILD_DIR']=str(engine.SOURCE/'tmp_build')
    gaussian=GaussianModel(parsed.sh_degree);scene=Scene(mp.extract(parsed),gaussian,load_iteration='50000',shuffle=False)
    # Fixed, already frozen global scene clock. Training knots only derive from train JSON.
    for views in [scene.getTrainCameras(),scene.getTestCameras()]:
        for view in views:view.fid=view.fid.new_tensor([(int(view.image_name)-1)/199])
    train=engine.read_json(ROOT/'data/derived/kinematic_resplit/training/transforms_train.json')['frames']
    times=sorted((int(Path(f['file_path']).stem)-1)/199 for f in train)
    assert len(times)==120
    deform=DeformModel(parsed.grid_args,parsed.network_args,scale_xyz=parsed.scale_xyz,reg_spatial_able=False,reg_temporal_able=False)
    deform.load_weights(RUN,iteration='50000')
    perceptual=lpips.LPIPS(net='vgg').cuda().eval().requires_grad_(False)
    meta=engine.read_json(source/'transforms_test.json')['frames'];frame_paths={Path(f['file_path']).stem:source/(f['file_path']+'.png') for f in meta}
    pipe=pp.extract(parsed);bg=torch.zeros(3,device='cuda');rows={k:[] for k in METHODS};views=scene.getTestCameras();started=time.time()
    if args.split=='test':
        selected=selection['selected_method'];methods={'native':METHODS['native'],selected:METHODS[selected]}
    else:methods=METHODS
    state={'status':'validation_readout_ablation' if args.split=='validation' else 'locked_test_export','completed':0,'total':len(views),'split':args.split,'methods':list(methods),'started_utc':datetime.now(timezone.utc).isoformat()}
    monitor(state)
    def query(x,t):return deform.step(x,x.new_full((len(x),1),t),fixed_attention=True)
    with torch.no_grad():
        x=gaussian.get_xyz.detach()
        for i,view in enumerate(views):
            view.load2device();t=float(view.fid.item());current=query(x,t);bounds=bracket(times,t)
            left,right,w=(query(x,bounds[0]),query(x,bounds[1]),bounds[2]) if bounds else (current,current,0.)
            target=read_alpha(frame_paths[view.image_name],(view.image_width,view.image_height)).numpy()
            gt=quantize_png(view.original_image[:3].cuda());filename=f'{i:05d}.png'
            save_tensor(output/'gt'/filename,gt)
            for name,(strength,full) in methods.items():
                pkg=repaired_state(x,gaussian.get_scaling,gaussian.get_rotation,current,left,right,w,strength if bounds else 0.,full)
                raster=render(view,gaussian,pipe,bg,pkg['d_xyz'],pkg['d_rotation'],pkg['d_scaling'])
                rgb=quantize_png(raster['render']);alpha=quantize_png(raster['alpha'])
                scores={'PSNR':psnr(rgb[None],gt[None]).item(),'SSIM':ssim(rgb[None],gt[None]).item(),
                    'LPIPS':perceptual(rgb[None],gt[None]).item(),'MS-SSIM':ms_ssim(rgb[None],gt[None],data_range=1,size_average=True).item()}
                integrity=image_metrics(alpha.cpu().numpy(),target,np.moveaxis(rgb.cpu().numpy(),0,-1),np.moveaxis(gt.cpu().numpy(),0,-1))
                rows[name].append({'index':i,'frame':view.image_name,'time':t,'training_bracket':bounds,'metrics':scores,'integrity':integrity})
                save_tensor(output/name/'renders'/filename,rgb);save_tensor(output/name/'alpha'/filename,alpha)
                del raster,rgb,alpha
            view.load2device('cpu');state.update(completed=i+1,elapsed_seconds=round(time.time()-started,1),last_frame=view.image_name);monitor(state)
            print('READOUT_FRAME '+json.dumps({'index':i,'frame':view.image_name,'PSNR':{k:v[-1]['metrics']['PSNR'] for k,v in rows.items() if v}}),flush=True)
    summary={}
    for name in methods:
        r=rows[name];summary[name]={'metrics':{k:math.fsum(x['metrics'][k] for x in r)/len(r) for k in engine.METRICS},
            'integrity':aggregate([x['integrity'] for x in r]),'worst5_PSNR':math.fsum(sorted(x['metrics']['PSNR'] for x in r)[:5])/5,'per_frame':r}
    result={'split':args.split,'training_knots':times,'train_targets_used':False,'trained_parameters_changed':False,
        'covariance_quaternion_magnitudes_preserved':True,'rendered':str(output),'models':summary}
    if args.split=='validation':
        expected=engine.read_json(ROOT/'experiments/integrity_continuation/assessment.json')['models']
        expected=next(x['validation']['metrics'] for x in expected if x['model']=='H3_combined')
        difference={k:summary['native']['metrics'][k]-expected[k] for k in engine.METRICS}
        if any(abs(x)>1e-4 for x in difference.values()):raise ValueError('Native H3 parity failed: '+str(difference))
        result['native_parity_delta']=difference
    write(REPORTS/('readout_'+args.split+'.json'),result)
    state.update(status='completed',summary={k:{j:v[j] for j in ('metrics','integrity','worst5_PSNR')} for k,v in summary.items()});monitor(state)


if __name__=='__main__':main()
