"""Figures and browser for the bounded motion correspondence test."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.motion_correspondence.run import OUT,REPORT,VARIANTS,read
from output.motion_correspondence.render_support import PALETTE,labelled


def main():
    controlled=read(REPORT/'controlled.json');synthetic=read(REPORT/'synthetic.json')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained');names=list(VARIANTS);x=np.arange(3)
    errors=[controlled['summary'][n]['forecast_error_mean_pixels'] for n in names]
    axes[0].bar(x,errors,color=['#888','#4c92c3','#39a485']);axes[0].set_xticks(x,names,rotation=12);axes[0].set(title='Artificial gaps: forecast before target',ylabel='mean pixel error (pseudo-reference)')
    correct=np.array([controlled['summary'][n]['correct'] for n in names]);wrong=np.array([controlled['summary'][n]['wrong'] for n in names]);unknown=np.array([controlled['summary'][n]['abstained'] for n in names])
    axes[1].bar(x,correct,label='accepted correct',color='#39a485');axes[1].bar(x,wrong,bottom=correct,label='accepted wrong',color='#d86261');axes[1].bar(x,unknown,bottom=correct+wrong,label='unknown',color='#aaa')
    axes[1].set_xticks(x,names,rotation=12);axes[1].set(title='Artificial gaps: accepted image identity',ylabel='cases against independent flow pseudo-reference');axes[1].legend(fontsize=8)
    for name in names:
        r=read(REPORT/(name+'.json'))['frames'];axes[2].plot([v['frame'] for v in r],[sum(j>=0 for j in v['accepted_assignment']) for v in r],label=name)
    axes[2].set(title='Natural sequence: acceptance, NOT accuracy',xlabel='original training frame',ylabel='accepted candidate IDs',ylim=(-.2,4.2));axes[2].legend(fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for j,scene in enumerate(synthetic):
        r=synthetic[scene];ax=axes[j]
        for i,name in enumerate(names):
            accepted=r[name]['accepted'];truth=r[name]['reference_assignment'];label=str(accepted)
            if truth is not None:label+='  wrong='+str(r[name]['accepted_wrong'])
            ax.text(.02,.78-.25*i,name+'\n'+label,transform=ax.transAxes,fontsize=10)
        ax.set(title=scene);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Pass and bounce have identical observed endpoints; guarded model must retain ambiguity',fontsize=10)
    fig.savefig(OUT/'synthetic_review.png',dpi=145);plt.close(fig)
    images=[Image.open(p).convert('RGB') for p in sorted((OUT/'render_support').glob('frame*.png'))]
    canvas=Image.new('RGB',(2400,265*len(images)),'#182a3d')
    for i,image in enumerate(images):canvas.paste(image,(0,i*265))
    canvas.save(OUT/'render_review.png')
    frames=[r['frame'] for r in read(REPORT/'history_guarded.json')['frames']]
    selected=[2,42,43,55,63,74,86,108,134,172,200]
    html='''<!doctype html><meta charset="utf-8"><title>Motion history correspondence test</title><style>body{background:#142538;color:#eee;font:16px system-ui;margin:24px}img{width:100%;margin:12px 0}a{color:#8ee}p{max-width:1200px}input{width:60%}</style><h1>History-based correspondence before shape training</h1><p>One bounded non-neural image-plane tracking prototype; ZERO Gaussian training. Candidate colors are image identities, not verified material IDs. Unknown stays grey. Score margins are not calibrated probabilities. Controlled masks use independent FB pseudo-reference; natural identity accuracy is unavailable.</p><a href="../../experiments/motion_correspondence/assessment.md">Assessment</a> | <a href="../../experiments/motion_correspondence/controlled.json">Controlled results</a> | <a href="../../experiments/motion_correspondence/independent_verification.json">Independent verification</a><img src="comparison.png"><img src="synthetic_review.png"><p><input id="slider" type="range" min="0" max="10" value="0"><b id="label"></b></p><img id="frame"><p>Columns: GT | old visible FB evidence | CV candidate | guarded candidate | complete C1 early patch contribution | C1 centers. Full120 image-plane states and masks are in the JSON/NPZ receipts; compositor review uses11 fixed train frames.</p><script>const frames=FRAMES;const s=document.querySelector('#slider');function show(){const f=frames[+s.value];document.querySelector('#label').textContent='Original training frame '+f;document.querySelector('#frame').src='render_support/frame'+String(f).padStart(4,'0')+'.png'}s.oninput=show;show();</script>'''.replace('FRAMES',json.dumps(selected))
    (OUT/'index.html').write_text(html,encoding='utf-8')
    print('Motion figures and browser written')


if __name__=='__main__':main()
