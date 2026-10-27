"""Read-only final50k assessment and aligned GT/G0/G1 comparison exports."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'runs/pidg_centroid_tuning'
REPORTS = ROOT/'experiments/centroid_tuning'
ANALYSIS = REPORTS/'analysis'
GALLERY = RUNS/'render_comparisons/G0_G1_final50000'
NAMES = ['G0_control','G1_centroid']
KEYS = ['PSNR','SSIM','LPIPS','MS-SSIM']
SOURCES = {}


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    assert read(RUNS/'controller.json')['status']=='completed'
    split = ROOT/'data/derived/kinematic_resplit/final_test'
    frames = read(split/'transforms_test.json')['frames']
    assert [int(Path(f['file_path']).stem) for f in frames]==list(range(1,201,4))
    models = []
    for name in NAMES:
        d=RUNS/name/'rubber_duck'
        assert read(d/'checkpoints/resume.json')['iteration']==50000
        m={'id':name,'directory':str(d),'scores':read(d/'results.json')['ours_50000'],
           'per_view':read(d/'per_view.json')['ours_50000'],
           'coverage':read(d/'coverage_metrics_test.json')['50000']['mean'],
           'centroid':read(d/'centroid_metrics_test.json')['50000']['mean'],
           'config':read(d/'metric_config.json'),'branch':read(d/'branch.json'),
           'selection':read(d/'metric_best.json'),
           'history':[json.loads(row) for row in (d/'metric_history.jsonl').read_text().splitlines()]}
        assert all(v['selection_split']=='validation' for v in m['selection'].values())
        for key in KEYS:
            assert len(m['per_view'][key])==50
            assert abs(math.fsum(m['per_view'][key].values())/50-m['scores'][key])<1e-5
        diagnostic=[]
        for i,frame in enumerate(frames):
            path=d/f'test/ours_50000/alpha/{i:05d}.png'
            with Image.open(path) as image:
                alpha=np.array(image.convert('L'),dtype=float)/255.;size=image.size
            target=split/(frame['file_path']+'.png')
            with Image.open(target) as image:
                gt=np.array(image.getchannel('A').resize(size,Image.Resampling.BICUBIC),dtype=float)/255.
            background=gt<=.5
            y,x=np.mgrid[:size[1],:size[0]]
            coords=np.stack((x/(size[0]-1),y/(size[1]-1)),-1)
            centers=[(a[...,None]*coords).mean((0,1))/max(float(a.mean()),1e-6) for a in (alpha,gt)]
            error=centers[0]-centers[1]
            diagnostic.append({'frame_id':int(Path(frame['file_path']).stem),
                'centroid_distance':float(np.linalg.norm(error)),
                'centroid_distance_pixels':float(np.linalg.norm(error*np.array([size[0]-1,size[1]-1]))),
                'background_false_coverage':float(((alpha>.5)&background).sum()/max(1,background.sum())),
                'predicted_foreground_fraction':float((alpha>.5).mean()),'gt_foreground_fraction':float((gt>.5).mean())})
        m['diagnostic']={'per_frame':diagnostic,'mean':{k:float(np.mean([r[k] for r in diagnostic])) for k in diagnostic[0] if k!='frame_id'}}
        assert abs(m['diagnostic']['mean']['centroid_distance']-m['centroid']['centroid_distance'])<1e-6
        models.append(m)
    assert models[0]['branch']['identical_full_checkpoint_sha256']==models[1]['branch']['identical_full_checkpoint_sha256']
    common=RUNS/'shared_prefix/rubber_duck/checkpoints/resume.pt'
    assert hashlib.sha256(common.read_bytes()).hexdigest()==models[0]['branch']['identical_full_checkpoint_sha256']
    assert [r for r in models[0]['history'] if r['iteration']<=3000]==[r for r in models[1]['history'] if r['iteration']<=3000]
    c0,c1=[dict(m['config']) for m in models]
    assert c0.pop('centroid_weight')==0. and c1.pop('centroid_weight')==1. and c0==c1
    assert c0['perceptual_weight']==0. and c0['shared_physics_time'] is False
    deltas={k:np.array([models[1]['per_view'][k][f'{i:05d}.png']-models[0]['per_view'][k][f'{i:05d}.png'] for i in range(50)]) for k in KEYS}
    comparison={k:{'aggregate_delta':models[1]['scores'][k]-models[0]['scores'][k],
        'mean_paired_delta':float(v.mean()),'median_paired_delta':float(np.median(v)),
        'improved_frames':int((v<0 if k=='LPIPS' else v>0).sum()),'min_delta':float(v.min()),'max_delta':float(v.max())} for k,v in deltas.items()}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'snapshot':'final50000','split':[120,30,50],
        'models':models,'comparison':comparison,'shared_prefix_verified':True,
        'PSNR_sensitivity_posthoc':{'first3_original_ids':[1,5,9],'first3_mean_delta':float(deltas['PSNR'][:3].mean()),
            'remaining47_mean_delta':float(deltas['PSNR'][3:].mean()),'not_an_alternative_primary_metric':True},
        'conclusion':'Mixed: better mean PSNR/MS-SSIM and 2D alpha/centroid metrics; worse SSIM/LPIPS. PSNR improves on23/50 frames with a negative median; large early-frame gains drive the mean.',
        'visual_observations':[
            'Original frame1: G0 omits the upper duck; G1 recovers it but with blur.',
            'Original frame49: G1 has merged/blurred duck shapes and obstacle-boundary defects; this is the largest PSNR regression.',
            'Original frame153: dark obstacle defects remain in both; center agreement does not remove every visible gap.'],
        'limitations':['Single scene/seed; CUDA nondeterminism persists after the shared branch',
            'Historical test exposure; no checkpoint or weight is reselected using this assessment',
            '2D whole-scene alpha centroid is not a physical mass center or proof of correct3D geometry']}
    ANALYSIS.mkdir(parents=True,exist_ok=True);GALLERY.mkdir(parents=True,exist_ok=True)
    def composite(i):
        filename=f'{i:05d}.png'; images=[]
        for m in models:
            raw=Path(m['directory'])/'test/ours_50000'
            with Image.open(raw/'gt'/filename) as im: gt=im.convert('RGB')
            with Image.open(raw/'renders'/filename) as im: render=im.convert('RGB')
            images.append((gt,render))
        if images[0][0].tobytes()!=images[1][0].tobytes():raise ValueError('GT correspondence mismatch')
        panes=[images[0][0],images[0][1],images[1][1]]
        w,h=panes[0].size
        canvas=Image.new('RGB',(w*3,h+60),'#edf2f7');draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',26)
        labels=['GT',f"G0 | PSNR {models[0]['per_view']['PSNR'][filename]:.3f}",f"G1 + centroid | PSNR {models[1]['per_view']['PSNR'][filename]:.3f}"]
        for j,(im,label) in enumerate(zip(panes,labels)):
            canvas.paste(im,(j*w,60));draw.text((j*w+16,16),f'Frame {4*i+1:04d} | '+label,font=font,fill='#17304a')
        target=GALLERY/filename;canvas.save(target,compress_level=3)
        with Image.open(target) as check:
            for j,im in enumerate(panes):
                assert check.crop((j*w,60,(j+1)*w,h+60)).tobytes()==im.tobytes()
        return {'index':i,'original_frame_id':4*i+1,'filename':filename,'delta_psnr':float(deltas['PSNR'][i])}
    with ThreadPoolExecutor(max_workers=3) as pool:gallery=list(pool.map(composite,range(50)))
    page='''<!doctype html><meta charset="utf-8"><title>GT / G0 / G1 final50k</title>
<style>body{background:#101725;color:white;font:16px system-ui;margin:24px}img{width:100%}a{color:#8de1bc}input{width:350px}</style>
<h1>GT / original PIDG control / PIDG + centroid</h1><p>Final50k, same50 test frames. Whole-scene2D alpha centroid; no physical mass claim.</p>
<input id="frame" type="range" min="0" max="49" value="38"><span id="label"></span> <a id="full" target="_blank">Open original PNG</a><img id="image">
<script>const records=RECORDS,f=document.getElementById('frame');function update(){let r=records[Number(f.value)];document.getElementById('image').src=r.filename;document.getElementById('full').href=r.filename;document.getElementById('label').textContent=`Original frame ${r.original_frame_id} | G1-G0 PSNR ${r.delta_psnr.toFixed(3)} dB`;}f.oninput=update;update();</script>'''
    (GALLERY/'index.html').write_text(page.replace('RECORDS',json.dumps(gallery)),encoding='utf-8')
    (GALLERY/'manifest.json').write_text(json.dumps({'count':50,'tags':['50000'],'all_gt_equal':True,'all_panels_pixel_verified':True,'frames':gallery},indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,key in zip(axes.flat,KEYS):
        for m,color in zip(models,['#4775b6','#149b83']):
            ax.plot([r['iteration']/1000 for r in m['history']],[r['metrics'][key] for r in m['history']],label=m['id'],c=color)
        ax.axvline(3,color='gray',ls='--',lw=.8);ax.set(xlabel='Training steps (k)',ylabel='Validation '+key);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ANALYSIS/'validation_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    ids=np.arange(1,201,4)
    for m,color in zip(models,['#4775b6','#149b83']):
        axes[0].plot(ids,[m['per_view']['PSNR'][f'{i:05d}.png'] for i in range(50)],label=m['id'],c=color)
    axes[0].set(ylabel='Test PSNR (dB)');axes[0].legend()
    axes[1].bar(ids,deltas['PSNR'],width=2.5,color=np.where(deltas['PSNR']>=0,'#149b83','#c7675b'))
    axes[1].axhline(0,c='gray',lw=.8);axes[1].set(xlabel='Original test frame ID',ylabel='G1 - G0 PSNR (dB)')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ANALYSIS/'test_per_frame.png',dpi=180);plt.close(fig)
    for index in [0,12,38]:
        with Image.open(GALLERY/f'{index:05d}.png') as im:
            gt=np.asarray(im.crop((0,60,1600,960))); yy,xx=np.where(gt.max(-1)>5)
            box=(max(0,int(xx.min())-25),max(0,int(yy.min())-25),min(1600,int(xx.max())+26),min(900,int(yy.max())+26))
            crops=[im.crop((j*1600+box[0],60+box[1],j*1600+box[2],60+box[3])) for j in range(3)]
        fig,axes=plt.subplots(1,3,figsize=(9,5))
        for ax,crop,label in zip(axes,crops,['GT','G0','G1 + centroid']):ax.imshow(crop);ax.set_title(label);ax.axis('off')
        fig.suptitle(f'Original frame {4*index+1}; one GT-defined crop shared across models')
        fig.tight_layout();fig.savefig(ANALYSIS/f'frame{4*index+1:04d}.png',dpi=160);plt.close(fig)
    result['sources']=SOURCES
    (REPORTS/'assessment.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# G0 / G1: final50k assessment','',result['conclusion'],'',
        '| Metric | G0 | G1 + centroid | G1-G0 | Better frames /50 |','|---|---:|---:|---:|---:|']
    for key in KEYS:
        lines.append(f"| {key} | {models[0]['scores'][key]:.7f} | {models[1]['scores'][key]:.7f} | {comparison[key]['aggregate_delta']:+.7f} | {comparison[key]['improved_frames']} |")
    lines+=['','| Diagnostic | G0 | G1 |','|---|---:|---:|']
    for key,label,scale,where in [('foreground_psnr','Foreground PSNR',1,'coverage'),('mask_iou','Mask IoU',1,'coverage'),
        ('interior_hole_rate','Interior low coverage %',100,'coverage'),('centroid_distance','Normalized centroid distance',1,'centroid')]:
        lines.append('| '+label+' | '+' | '.join(f'{m[where][key]*scale:.7f}' for m in models)+' |')
    for key,label in [('centroid_distance_pixels','Centroid distance in pixels'),('background_false_coverage','Background false coverage %')]:
        scale=100 if key=='background_false_coverage' else 1
        lines.append('| '+label+' | '+' | '.join(f'{m["diagnostic"]["mean"][key]*scale:.5f}' for m in models)+' |')
    lines+=['',f"PSNR median paired delta: {comparison['PSNR']['median_paired_delta']:+.5f} dB. Only23/50 frames improve.",
        f"Post hoc sensitivity: first3 frames (IDs1,5,9) average {deltas['PSNR'][:3].mean():+.4f} dB; remaining47 average {deltas['PSNR'][3:].mean():+.4f} dB. This does not replace the primary50-frame score.",
        '', 'The center target and coverage improve, but SSIM/LPIPS worsen. A global alpha moment constrains overall distribution, not local shape, color, object identity, depth or physical mass. Large early-frame gains drive mean PSNR; the method is not uniformly better.',
        '', '## Visual observations','', *[f'- {text}' for text in result['visual_observations']],
        '', 'Both runs share an identical complete3k state, confirmed by checkpoint SHA and identical pre-activation histories. Only centroid_weight differs. Single-scene/seed and post-branch CUDA divergence remain limitations. No test-based reselection is performed.',
        '', '## Figures','', '![Validation](analysis/validation_comparison.png)','', '![Per-frame test](analysis/test_per_frame.png)',
        '', '## GT / G0 / G1 renders','',f'Combined50-frame gallery: `{GALLERY / "index.html"}`',
        '', 'Each source panel is preserved at1600x900 and pixel-verified; all GT pairs match.',
        '', *[f'![Original frame {i:04d}](analysis/frame{i:04d}.png)\n' for i in [1,49,153]]]
    (REPORTS/'assessment.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')
    path=REPORTS/'results.md';body=path.read_text(encoding='utf-8').split('\n## Verified final assessment')[0]
    path.write_text(body+'\n## Verified final assessment\n\n[Full assessment and plots](assessment.md)\n\n'+result['conclusion']+'\n',encoding='utf-8')
    print(json.dumps({'comparison':comparison,'diagnostics':{m['id']:m['diagnostic']['mean'] for m in models},'gallery':str(GALLERY/'index.html')},indent=2))


if __name__=='__main__':main()
