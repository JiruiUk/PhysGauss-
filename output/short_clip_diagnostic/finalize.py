"""Figures and browser for the fixed-clip diagnostic; no model execution."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.short_clip_diagnostic.image_camera import REPORT,OUT,TRAIN,read,write,sha,image


def main():
    model=read(REPORT/'model_results.json');camera=read(REPORT/'image_camera.json');native=read(REPORT/'resolution_control.json');gates=read(REPORT/'gate_analysis.json')
    names=[c['clip'] for c in model['clips']];labels=['2–14','188–200','15–27','2–8*'];x=np.arange(4)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    values=[[c['mean_forward_accepted_duck_alpha_coverage'] for c in model['clips']],
       [c['comparison']['same400_banks']['mean_forward_alpha_coverage'] for c in native['full_clip_comparisons']],
       [c['comparison']['native_seed_banks']['mean_forward_alpha_coverage'] for c in native['full_clip_comparisons']]]
    for i,(v,label) in enumerate(zip(values,['400 seed /400 render','same400 seed /1600 render','1600 seed /1600 render'])):
        axes[0,0].bar(x+(i-1)*.25,np.array(v)*100,width=.25,label=label)
    axes[0,0].set_xticks(x,labels);axes[0,0].set(title='Fixed bank support, forward observations only',ylabel='accepted visible duck alpha (%)',ylim=(0,100));axes[0,0].legend(fontsize=8)
    for i,(key,label) in enumerate([('original_joint_mean_alpha_coverage','cohort + point'),('point_only_mean_alpha_coverage','same point rules, no cohort rejection'),('raw_bank_mean_alpha_coverage','raw bank mass on target')]):
        axes[0,1].bar(x+(i-1)*.25,[c[key]*100 for c in gates['clips']],width=.25,label=label)
    axes[0,1].set_xticks(x,labels);axes[0,1].set(title='400 cohort-gate sensitivity; post-result diagnostic',ylabel='visible duck alpha (%)',ylim=(0,100));axes[0,1].legend(fontsize=8)
    for method,ls in [('raw_blue','--'),('LK_blue','-')]:
        for key,color,title in [('legacy__legacy','#cf766a','trained camera'),('metadata_cv__metadata','#379b91','metadata convention hypothesis')]:
            v=[c['camera_scores'][method][key]['median_of_pair_medians_pixels'] for c in camera['clips']]
            axes[1,0].plot(x,v,ls=ls,marker='o',color=color,label=method+' '+title)
    axes[1,0].set_xticks(x,labels);axes[1,0].set(title='Conditional static-blue epipolar test; no camera GT',ylabel='Sampson distance, px',ylim=(0,1));axes[1,0].legend(fontsize=7)
    for offset,(key,label) in enumerate([('direct400_vs_trainingStyleGT_PSNR','direct400'),('native1600_vs_trainingGT_PSNR','trained1600')]):
        v=[np.mean([r[key] for r in native['RGB'] if r['frame'] in c['frames']]) for c in model['clips']]
        axes[1,1].bar(x+(offset-.5)*.3,v,width=.3,label=label)
    axes[1,1].set_xticks(x,labels);axes[1,1].set(title='Rendering-resolution control, NOT model improvement',ylabel='training RGB diagnostic PSNR (dB)',ylim=(20,60));axes[1,1].legend(fontsize=8)
    fig.suptitle('Four fixed clips, frozen C1; *2–8 is nested in2–14, not independent',fontsize=12)
    fig.savefig(OUT/'comparison.png',dpi=145);plt.close(fig)
    # Native-resolution endpoint gates, with SAME400-selected rows to isolate
    # rendering resolution; show thresholds without turning them into GT.
    late=next(c for c in native['endpoints'] if c['target']==200)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for j,(key,title,threshold) in enumerate([('footprint_purity','Footprint purity',95),('mass_weighted_center_inside','Contribution-weighted centers inside',90)]):
        for i,mode in enumerate(['direct400','native1600_normalized']):
            axes[j].bar(np.arange(4)+(i-.5)*.3,[b[mode]['score'][key]*100 for b in late['banks']],width=.3,label=mode)
        axes[j].axhline(threshold,color='#d15a5a',ls='--',label=f'gate{threshold}%');axes[j].set_xticks(range(4),['F0','F1','F2','F3']);axes[j].set(title=title+' at200',ylim=(75,102),ylabel='%');axes[j].legend(fontsize=8)
    fig.savefig(OUT/'late_gates.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for clip in model['clips']:
        if clip['clip'].endswith('0002_0008'):continue
        rr=[r for r in model['bank_frames'] if r['clip']==clip['clip'] and r['bank'].endswith('blue') and r['phase']=='forward']
        axes[0].plot([r['frame'] for r in rr],[r['world_center_displacement_from_seed_median_scene_units'] for r in rr],marker='o',label=clip['clip'])
        axes[1].plot([r['frame'] for r in rr],[r['motion_split']['full_projection_weighted_EPE'] for r in rr],marker='o',label=clip['clip']+' full')
        axes[1].plot([r['frame'] for r in rr],[r['motion_split']['camera_only_weighted_EPE'] for r in rr],ls='--',label=clip['clip']+' camera only')
    axes[0].set(title='Blue model-row world drift; static-blue assumption',xlabel='original frame',ylabel='median displacement, scene units');axes[0].legend(fontsize=7)
    axes[1].set(title='Projection decomposition depends on C1 source depth',xlabel='original frame',ylabel='raw-flow EPE, px');axes[1].legend(fontsize=6)
    fig.savefig(OUT/'blue_motion.png',dpi=145);plt.close(fig)
    # Contact sheets of true1600 training-style GT|render (display half-size).
    endpoints=[c['frames'][-1] for c in model['clips']];sheet=Image.new('RGB',(3200,980),'#182a3d');draw=ImageDraw.Draw(sheet)
    for i,n in enumerate(endpoints):
        with Image.open(OUT/'native_gt_render'/f'frame{n:04d}.png') as im:small=im.resize((1600,450),Image.Resampling.LANCZOS)
        xx=(i%2)*1600;yy=(i//2)*490;sheet.paste(small,(xx,yy+40));draw.text((xx+8,yy+10),f'Original training frame{n}; GT LEFT | frozen C1 RIGHT, original1600 render',fill='white')
    sheet.save(OUT/'native_review.png')
    metadata=read(TRAIN/'transforms_train.json');lookup={int(Path(f['file_path']).stem):f for f in metadata['frames']}
    with np.load(OUT/'image_correspondences.npz') as z:points={k:z[k] for k in z.files}
    sheet=Image.new('RGB',(800,265*4),'#182a3d');draw=ImageDraw.Draw(sheet)
    for i,c in enumerate(model['clips']):
        a,b=c['frames'][-2:];first,_=image(lookup[a]);second,_=image(lookup[b]);panel=Image.fromarray(np.concatenate([first,second],1));dr=ImageDraw.Draw(panel)
        for region,color in [('LK_blue',(40,220,220)),('duck_LK',(245,160,80))]:
            aa=points[f'{a}_{b}_{region}_source'];bb=points[f'{a}_{b}_{region}_target']
            for p,q in zip(aa,bb):
                x0,y0=p;x1,y1=q;dr.ellipse((x0-1,y0-1,x0+1,y0+1),outline=color);dr.ellipse((400+x1-1,y1-1,400+x1+1,y1+1),outline=color)
        sheet.paste(panel,(0,i*265+40));draw.text((6,i*265+10),f'{a} ->{b}: independent LK; cyan blue structures, orange duck regions',fill='white')
    sheet.save(OUT/'LK_review.png')
    # Cache receipts bind numerical evidence without changing historical runs.
    cache=ROOT/'runs/pidg_short_clip_diagnostic/cache'
    write(REPORT/'cache_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(cache.glob('*.npz'))})
    html='''<!doctype html><meta charset="utf-8"><title>Four-clip C1 diagnostic</title><style>body{background:#142538;color:#eee;font:16px system-ui;margin:24px}img{max-width:100%}a{color:#8ee}</style><h1>Fixed clear clips: image tracking, cameras, C1 surface support</h1><p>Frozen C1, no optimizer updates. Original1600 rendering and400 diagnostics differ; coverage is not geometry accuracy. 2–8 is nested in2–14. Camera tests assume blue structures static and use estimated image correspondences, not3D GT.</p><a href="../../experiments/short_clip_diagnostic/assessment.md">Assessment</a> | <a href="../../experiments/short_clip_diagnostic/verification.json">Independent verification</a><img src="comparison.png"><img src="late_gates.png"><img src="blue_motion.png"><img src="native_review.png"><img src="LK_review.png"><h2>Native GT|Render endpoints</h2>'''
    for n in endpoints:html+=f'<p>Original frame{n}; GT LEFT | Render RIGHT</p><img src="native_gt_render/frame{n:04d}.png">'
    html+='<h2>All fixed observations, trained1600 GT|Render</h2><p>First three observations select local rows; later observations evaluate them. These are training views, not a new heldout benchmark.</p>'
    for c in model['clips']:
        html+=f'<h3>{c["clip"]}</h3><p>'
        for i,n in enumerate(c['frames']):
            phase='seed' if i<3 else 'forward'
            html+=f'<a href="native_gt_render/frame{n:04d}.png">{n} ({phase})</a> &nbsp; '
        html+='</p>'
    (OUT/'index.html').write_text(html,encoding='utf-8');print('SHORT CLIP DIAGNOSTIC FIGURES COMPLETE',flush=True)


if __name__=='__main__':main()
