"""Static figures and a local full-sequence browser from audit outputs."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.surface_support.audit import OUT,REPORT,TRAIN,PALETTE,rgba
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    meta=read(TRAIN/'transforms_train.json');frames={int(Path(f['file_path']).stem):f for f in meta['frames']}
    data={n:read(REPORT/(n+'.json')) for n in ['C1_final','P1_16k']}
    tracking=read(REPORT/'image_tracking.json')['frames']
    fig,axes=plt.subplots(2,3,figsize=(16,8.5),constrained_layout=True)
    for row,(name,m) in enumerate(data.items()):
        r=m['frames'];t=[x['frame'] for x in r]
        ax=axes[row,0]
        for key,label,style in [('exact_known_share','early patches: full compositor','-'),('old_first20_normalized_known_share','first20 normalized estimate',':'),('late86_local_probe_exact_share','late86 probes: full compositor','--')]:
            ax.plot(t,[x[key] for x in r],style,label=label)
        ax.set(title=name+' | visible duck contribution',ylabel='share of rendered alpha on yellow ROI',ylim=(0,1));ax.legend(fontsize=8)
        ax=axes[row,1]
        for key,label in [('known_centers_in_frustum_fraction','centers in view'),('known_centers_on_duck_fraction','centers on yellow ROI')]:ax.plot(t,[x[key] for x in r],label=label)
        ax.set(title='Center support vs actual footprint',ylabel='fraction of early marked kernels',ylim=(-.03,1.03));ax.legend(fontsize=8)
        ax=axes[row,2]
        for key,label in [('actual_known_duck_mass','actual'),('isolated_known_duck_mass','other kernels removed'),('fixed_clear_footprint_duck_mass','isolated + fixed clear footprint')]:ax.plot(t,[x[key]/x['duck_pixels'] for x in r],label=label)
        ax.set(title='Footprint / occlusion counterfactuals',ylabel='mean cohort opacity on yellow ROI');ax.legend(fontsize=8)
        for ax in axes[row]:
            ax.axvline(43,c='grey',lw=.8,ls=':');ax.axvline(86,c='grey',lw=.8,ls=':');ax.set_xlabel('Original training frame ID');ax.grid(alpha=.2)
    fig.savefig(OUT/'comparison.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    for k in range(4):axes[0].plot([r['frame'] for r in tracking],[r['tracked_pixels'][k] for r in tracking],color=PALETTE[k],label=f'early image patch {k}')
    axes[0].axvline(43,c='grey',ls=':');axes[0].set(title='Conservative visible-only raw FB tracking',xlabel='Original training frame ID',ylabel='remaining tracked pixels');axes[0].legend(fontsize=8)
    for name,m in data.items():axes[1].plot([r['frame'] for r in m['frames']],[r['first20_total_share'] for r in m['frames']],label=name)
    axes[1].set(title='First20 metadata coverage',xlabel='Original training frame ID',ylabel='metadata alpha / full compositor alpha',ylim=(0,1));axes[1].legend()
    fig.savefig(OUT/'correspondence_and_metadata.png',dpi=140);plt.close(fig)
    fig=plt.figure(figsize=(13,6),layout='constrained')
    overlays=[];contact=[]
    for j,(name,m) in enumerate(data.items()):
        a=np.load(OUT/name/'trajectories.npz');t=a['frames'];labels=a['labels'];ax=fig.add_subplot(1,2,j+1,projection='3d')
        for k in range(4):
            pool=np.where(labels==k)[0];chosen=pool[np.linspace(0,len(pool)-1,min(8,len(pool))).round().astype(int)]
            for n in chosen:
                points=a['centers'][:,n];ax.plot(*points.T,color=PALETTE[k],lw=.7,alpha=.6);ax.scatter(*points[0],color=PALETTE[k],s=8)
        # The black curves are a different, late-frame support cohort, not IDs.
        late=a['late86_probe_centers'];chosen=np.linspace(0,late.shape[1]-1,min(8,late.shape[1])).round().astype(int)
        for n in chosen:ax.plot(*late[:,n].T,c='black',ls='--',lw=.7,alpha=.5)
        ax.set(title=name+' | full center trajectories',xlabel='world X',ylabel='world Y',zlabel='world Z')
        for number in [2,74,86,200]:
            i=int(np.where(t==number)[0][0]);r=rgba(frames[number]);image=Image.fromarray(r[:,:,:3]).convert('RGB');draw=ImageDraw.Draw(image)
            for uv,k in zip(a['projected_centers'][i],labels):
                x,y=uv
                if np.isfinite([x,y]).all() and 0<=x<400 and 0<=y<225:draw.ellipse((x-.6,y-.6,x+.6,y+.6),fill=tuple((PALETTE[k]*255).astype(int)))
            canvas=Image.new('RGB',(400,255),'#182a3d');canvas.paste(image,(0,30));ImageDraw.Draw(canvas).text((5,8),f'{name} frame{number}: early centers on GT',fill='white');overlays.append(canvas)
        for number in [2,42,43,62,74,86,108,200]:contact.append(Image.open(OUT/name/f'frame{number:04d}.png').convert('RGB'))
    fig.suptitle('8 evenly spaced kernel indices per early patch; 8 late86 probes (black dashed). Not physical GT.',fontsize=10)
    fig.savefig(OUT/'center_trajectories_3d.png',dpi=150);plt.close(fig)
    canvas=Image.new('RGB',(1600,510),'#182a3d')
    for i,image in enumerate(overlays):canvas.paste(image,((i%4)*400,(i//4)*255))
    canvas.save(OUT/'center_review.png')
    canvas=Image.new('RGB',(2000,265*len(contact)),'#182a3d')
    for i,image in enumerate(contact):canvas.paste(image,(0,i*265))
    canvas.save(OUT/'sequence_review.png')
    ids=[int(Path(f['file_path']).stem) for f in meta['frames']]
    html='''<!doctype html><meta charset="utf-8"><title>Surface support audit</title><style>body{background:#142538;color:#eee;font:16px system-ui;margin:24px}img{width:100%;margin:8px 0}input{width:65%}a{color:#8ee}p{max-width:1100px}</style>
<h1>Surface support before shape constraints</h1><p>Read-only: C1 final50k and P1 stopped16k before shape updates. All 120 TRAIN frames. No new model training. Early patch colors are not material IDs; black or untracked regions remain unknown. The fourth image encodes R actual, G isolated, B isolated fixed-clear-footprint contribution. Fifth: conservative estimated FB visible tracking, no re-ID across ambiguous gaps.</p>
<a href="../../experiments/surface_support/assessment.md">Assessment</a> | <a href="../../experiments/surface_support/verification.json">Input integrity</a> | <a href="../../experiments/surface_support/independent_verification.json">Independent checks</a>
<p><input id="slider" type="range" min="0" max="119" value="0"><b id="label"></b> <button onclick="set(2)">Early 2</button> <button onclick="set(43)">Merge 43</button> <button onclick="set(74)">74</button> <button onclick="set(86)">86</button> <button onclick="set(200)">Late 200</button></p>
<img id="c1"><img id="p1"><img src="comparison.png"><img src="center_review.png"><img src="correspondence_and_metadata.png"><img src="center_trajectories_3d.png"><script>const frames=FRAMES;const s=document.querySelector('#slider');function show(){const f=frames[+s.value];document.querySelector('#label').textContent='Original training frame '+f;document.querySelector('#c1').src='C1_final/frame'+String(f).padStart(4,'0')+'.png';document.querySelector('#p1').src='P1_16k/frame'+String(f).padStart(4,'0')+'.png'}function set(f){s.value=frames.indexOf(f);show()}s.oninput=show;show();</script>'''.replace('FRAMES',json.dumps(ids))
    (OUT/'index.html').write_text(html,encoding='utf-8')
    print('Figures and full 120-frame browser written')


if __name__=='__main__':main()
