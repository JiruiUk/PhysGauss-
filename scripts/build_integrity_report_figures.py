"""Static, reproducible figures from audited final50k exports; no GPU work."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import gaussian_filter

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'output/pdf/integrity_analysis_assets'
PALETTE={'C1':'#64748b','H1':'#d58b26','H2':'#6b68ac','H3':'#008779','H4':'#b5629b','F1':'#d65b4b','F2':'#357ca5'}


def build():
    d=json.loads((A/'data.json').read_text(encoding='utf-8'));models={m['id']:m for m in d['models']}
    ids=list(models);base=models['C1']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190,'font.family':'DejaVu Sans'})
    fig,ax=plt.subplots(1,2,figsize=(10.5,3.6),layout='constrained')
    for a,metric,title in [(ax[0],'PSNR','PSNR change vs C1 (dB, higher better)'),(ax[1],'LPIPS','LPIPS change vs C1 (%, lower better)')]:
        vals=[models[k]['test'][metric]-base['test'][metric] for k in ids]
        if metric=='LPIPS':vals=[v/base['test'][metric]*100 for v in vals]
        a.barh(ids[::-1],vals[::-1],color=[PALETTE[k] for k in ids[::-1]])
        a.axvline(0,color='#64748b',lw=.7);a.set_title(title,fontsize=11)
        for y,v in enumerate(vals[::-1]):a.text(v+(.04 if v>=0 else -.04)*(1 if metric=='PSNR' else 7),y,f'{v:+.2f}',ha='left' if v>=0 else 'right',va='center',fontsize=9)
        lo,hi=min(vals),max(vals);a.set_xlim(lo-(hi-lo)*.24,hi+(hi-lo)*.28)
    fig.savefig(A/'quality.png',facecolor='white');plt.close(fig)
    x=np.array(d['protocol']['frame_ids']);fig,ax=plt.subplots(2,1,figsize=(10.3,5.4),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.4,1]})
    for key in ('C1','H3','F1'):
        ax[0].plot(x,[models[key]['per_view']['PSNR'][f'{i:05d}.png'] for i in range(50)],label=key,color=PALETTE[key],lw=1.3)
    delta=np.array([r['PSNR_delta'] for r in d['pairwise_vs_C1']['H3']['per_frame']])
    ax[1].bar(x,delta,width=2.5,color=np.where(delta>=0,PALETTE['H3'],PALETTE['F1']))
    ax[1].axhline(0,color='#64748b',lw=.8);ax[0].set_ylabel('PSNR (dB)');ax[1].set_ylabel('H3 - C1 (dB)');ax[1].set_xlabel('Global frame ID (1, 5, ..., 197)')
    ax[0].legend(ncol=3,loc='lower right',frameon=False);ax[0].set_title('Fixed test: 50 time-disjoint, interleaved frames',fontsize=12)
    for idx in (18,21):ax[1].annotate(str(x[idx]),(x[idx],delta[idx]),xytext=(x[idx]+6,delta[idx]),fontsize=9)
    fig.savefig(A/'time.png',facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.2,4.0),layout='constrained')
    for key,m in models.items():
        rows=[r for r in m['history'] if r['iteration']>=10000]
        unique={r['iteration']:r for r in rows};rows=[unique[i] for i in sorted(unique)]
        ax.plot([r['iteration']/1000 for r in rows],[r['metrics']['PSNR'] for r in rows],label=key,color=PALETTE[key],lw=2.1 if key=='H3' else 1.1,alpha=1 if key in ('H3','C1','F1') else .7)
    ax.set_xlabel('Training iteration (k)');ax.set_ylabel('Validation PSNR (dB)');ax.legend(ncol=7,fontsize=9,loc='upper left',frameon=False)
    ax.set_title('Validation only: 30 frames; screening at 20k is not the final ranking',fontsize=11)
    fig.savefig(A/'validation.png',facecolor='white');plt.close(fig)
    with (A/'per_frame_test.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['frame_id',*[k+'_PSNR' for k in ids],'H3_minus_C1'])
        for i,t in enumerate(x):w.writerow([int(t),*[models[k]['per_view']['PSNR'][f'{i:05d}.png'] for k in ids],float(delta[i])])
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    previews=[];crop_records=[]
    for i in (18,21):
        file=f'{i:05d}.png';gt=Image.open(Path(base['directory'])/'test/ours_50000/gt'/file).convert('RGB')
        images=[gt]+[Image.open(Path(models[k]['directory'])/'test/ours_50000/renders'/file).convert('RGB') for k in ('C1','H3','F1')]
        preview=Image.new('RGB',(1600,970),'white');p=ImageDraw.Draw(preview)
        for j,(label,im) in enumerate(zip(('GT','C1','H3','F1'),images)):
            xx=(j%2)*800;yy=(j//2)*485;p.text((xx+12,yy+4),f'{label} / frame {x[i]}',font=font,fill='#153149');preview.paste(im.resize((800,450)),(xx,yy+35))
        preview.save(A/f'preview_{x[i]}.jpg',quality=92)
        # Pick the largest local error difference, within visible GT foreground.
        a,b,c=[np.asarray(im,dtype=np.float32)/255 for im in images[:3]]
        diff=((b-a)**2).mean(-1)-((c-a)**2).mean(-1)
        if i==21:diff=-diff
        score=gaussian_filter(diff,25);score[(a.max(-1)<.08)]=-1
        yy,xx=np.unravel_index(score.argmax(),score.shape)
        crop_w,crop_h=380,330
        xx=int(np.clip(xx-crop_w//2,0,1600-crop_w));yy=int(np.clip(yy-crop_h//2,0,900-crop_h));box=(xx,yy,xx+crop_w,yy+crop_h)
        panel=Image.new('RGB',(4*crop_w,crop_h+42),'white');p=ImageDraw.Draw(panel)
        for j,(label,im) in enumerate(zip(('GT','C1','H3','F1'),images)):
            p.text((j*crop_w+12,7),label,font=font,fill=PALETTE.get(label,'#153149'));panel.paste(im.crop(box),(j*crop_w,42))
        panel.save(A/f'crop_{x[i]}.png')
        crop_records.append({'frame_id':int(x[i]),'test_index':i,'box_xyxy':box,'criterion':'max local squared-error advantage' if i==18 else 'max local squared-error disadvantage','H3_minus_C1':float(delta[i])})
        for im in images:im.close()
    (A/'crops.json').write_text(json.dumps(crop_records,indent=2),encoding='utf-8')
    print(json.dumps({'figures':['quality','time','validation'],'crops':crop_records},indent=2))


if __name__=='__main__':build()
