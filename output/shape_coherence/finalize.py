"""Post-selection comparison only. Historical C1 is read-only; no new training."""
from pathlib import Path
import sys
import os
import json
from datetime import datetime,timedelta,timezone
import numpy as np
import torch
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.shape_coherence.run import RUNS,REPORTS,OUT,NAME,CONTROL,prepare,wire,assert_frozen
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from output.shape_coherence.continue_run import control_unchanged
from output.c1_flow_ablation.compare_reuse import with_velocity_default
NAMES=['C1_repaired',NAME]


@torch.no_grad()
def contact_roi(m,directory,tag):
    from variants.shape_coherence.observations import camera,PIPE
    from variants.flow_comparison.temporal import warp_at
    from gaussian_renderer import render
    os.environ['PIDG_METRIC_CONFIG']=str(directory/'metric_config.json')
    metadata=engine.read_json(m.TRAIN/'transforms_test.json');g,d=m.load(directory,tag);rows=[]
    for frame in metadata['frames']:
        number=int(Path(frame['file_path']).stem)
        if not 62<=number<=108:continue
        cam=camera(frame,metadata,800);pkg=warp_at(d,g.get_xyz.detach(),(number-1)/199)
        image=render(cam,g,PIPE,m.BACKGROUND,pkg['d_xyz'],pkg['d_rotation'],pkg['d_scaling'])['render']
        with Image.open(m.TRAIN/(frame['file_path']+'.png')) as im:
            rgba=np.asarray(im.convert('RGBA').resize((800,450),Image.Resampling.BILINEAR))/255
        gt=rgba[...,:3]*rgba[...,3:4]
        mask=(rgba[...,3]>.95)&(gt[...,0]>.25)&(gt[...,1]>.18)&(gt[...,2]<.8*np.minimum(gt[...,0],gt[...,1]))
        mask=binary_dilation(mask,iterations=4)
        assert mask.sum()>0
        error=np.abs(image.permute(1,2,0).clamp(0,1).cpu().numpy()-gt).mean(-1)
        rows.append({'frame':number,'pixels':int(mask.sum()),'MAE':float(error[mask].mean())})
    assert rows
    del g,d;torch.cuda.empty_cache()
    return {'mean_frame_MAE':float(np.mean([r['MAE'] for r in rows])),'per_frame':rows,
        'split':'validation','width':800,'dilation_pixels':4,'window':[62,108]}


def review_panels():
    roots=[ROOT/'output/c1_flow_ablation/C1_repaired',OUT/NAME]
    canvas=Image.new('RGB',(1600,1000),'#17283b');draw=ImageDraw.Draw(canvas)
    for col,(name,root) in enumerate(zip(NAMES,roots)):
        for row,index in enumerate([0,21,25,49]):
            with Image.open(root/'pairs/best'/f'{index:05d}.png') as im:preview=im.convert('RGB').resize((800,238))
            canvas.paste(preview,(col*800,row*250));draw.text((col*800+8,row*250+238),f'{name} | index {index}',fill='white')
    canvas.save(OUT/'png_review.png')
    import imageio_ffmpeg,subprocess
    decoded=Image.new('RGB',(1600,1000),'#17283b')
    for col,root in enumerate(roots):
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(root/'GT_Render_best.mp4'),
                 '-vf','scale=800:238','-f','rawvideo','-pix_fmt','rgb24','-']
        result=subprocess.run(command,capture_output=True);assert result.returncode==0,result.stderr
        assert len(result.stdout)==50*800*238*3
        frames=np.frombuffer(result.stdout,dtype=np.uint8).reshape(50,238,800,3)
        for row,index in enumerate([0,21,25,49]):decoded.paste(Image.fromarray(frames[index]),(col*800,row*250))
    decoded.save(OUT/'decoded_video_review.png')


def main():
    repo=prepare();wire(repo);assert_frozen(repo);control_unchanged()
    path=RUNS/NAME/'rubber_duck';receipt=engine.read_json(path/'checkpoints/resume.json')
    assert receipt['iteration']==50000 and receipt['sha256']==engine.sha256(path/'checkpoints/resume.pt')
    gate=engine.read_json(REPORTS/'candidate_graph_review.json');assert gate['approved']
    expected={'selection_split':'validation','historical_control_reused':True,'models':{
        NAMES[0]:engine.read_json(CONTROL/'metric_best.json')['PSNR'],
        NAME:engine.read_json(path/'metric_best.json')['PSNR']}}
    lock=RUNS/'selection_lock.json'
    if lock.exists():assert all(engine.read_json(lock)[k]==v for k,v in expected.items())
    else:engine.write_json(lock,{**expected,'locked_utc':engine.now()})
    with engine.lock():
        state={'status':'rendering','pid':os.getpid(),'updated_utc':engine.now()}
        engine.write_json(RUNS/'controller.json',state)
        try:
            base.render(path,repo,datetime.now(timezone.utc)+timedelta(hours=6),tags=['best','50000'])
            from variants.flow_comparison import export as exporter
            exporter.RUNS,exporter.REPORTS,exporter.OUT=RUNS,REPORTS,OUT
            exporter.read=with_velocity_default(exporter.read);exporter.export(path)
            from variants.c1_flow_ablation.results import motion_module
            m=motion_module();m.read=with_velocity_default(m.read)
            manifest=m.read(m.CACHE/'manifest.json')
            assert all(m.sha(m.CACHE/k)==v for k,v in manifest['output_sha256'].items())
            rows=[m.evaluate(directory,tag,manifest) for directory in [CONTROL,path] for tag in ['best','50000']]
            engine.write_json(REPORTS/'motion_validation.json',{'results':rows,
                'observation_manifest_sha256':m.sha(m.CACHE/'manifest.json'),'not_3D_ground_truth':True})
            roi={directory.parent.name:{tag:contact_roi(m,directory,tag) for tag in ['best','50000']} for directory in [CONTROL,path]}
            validation={directory.parent.name:engine.read_json(directory/'metric_best.json')['PSNR']['metrics'] for directory in [CONTROL,path]}
            test={directory.parent.name:engine.read_json(directory/'results.json') for directory in [CONTROL,path]}
            delta={'validation_PSNR_dB':validation[NAME]['PSNR']-validation[NAMES[0]]['PSNR'],
                'validation_LPIPS_relative':validation[NAME]['LPIPS']/validation[NAMES[0]]['LPIPS']-1,
                'validation_contact_ROI_MAE_relative':roi[NAME]['best']['mean_frame_MAE']/roi[NAMES[0]]['best']['mean_frame_MAE']-1}
            criteria={'image_PSNR':delta['validation_PSNR_dB']>=-.2,'image_LPIPS':delta['validation_LPIPS_relative']<=.05,
                      'contact_ROI':delta['validation_contact_ROI_MAE_relative']<=-.10}
            result={'validation':validation,'test':test,'contact_ROI':roi,'deltas_P1_minus_C1':delta,
                'numeric_gates':criteria,'numeric_gates_passed':all(criteria.values()),'visual_gate_pending':True,
                'composite_benefit':None,'limitations':engine.read_json(REPORTS/'plan.json')['limits']}
            engine.write_json(REPORTS/'results.json',result);review_panels()
            page='<!doctype html><meta charset="utf-8"><title>C1 / P1 shape comparison</title><style>body{background:#142638;color:white;font:16px system-ui;margin:20px}td{width:50%}video,img{width:100%}pre{white-space:pre-wrap}</style><h1>C1 与局部形态候选 P1</h1><p>历史 C1 复用，未重训。每幅左GT、右Render。视频10fps仅为播放速度。</p><button onclick="v0.currentTime=v1.currentTime=0;v0.play();v1.play()">同步播放</button><button onclick="v0.pause();v1.pause()">暂停</button><table><tr>'
            roots=['../c1_flow_ablation/C1_repaired',NAME]
            for i,(name,root) in enumerate(zip(NAMES,roots)):
                page+=f'<td><h2>{name}</h2><video id="v{i}" controls loop src="{root}/GT_Render_best.mp4"></video><img id="im{i}"></td>'
            page+='</tr></table><input id="f" type="range" min="0" max="49" value="21"><pre>'+json.dumps(result,ensure_ascii=False,indent=2)+'</pre><script>const roots='+json.dumps(roots)+';function u(){roots.forEach((r,i)=>document.getElementById("im"+i).src=r+"/pairs/best/"+String(f.value).padStart(5,"0")+".png")}f.oninput=u;u()</script>'
            (OUT/'index.html').write_text(page,encoding='utf8')
            from output.shape_coherence.verify_results import main as verify
            verify()
            state['status']='completed_pending_manual_review'
        except Exception as error:
            state.update(status='postprocess_failed',error=str(error));raise
        finally:
            state['updated_utc']=engine.now();engine.write_json(RUNS/'controller.json',state)
    control_unchanged();assert_frozen(repo)


if __name__=='__main__':main()
