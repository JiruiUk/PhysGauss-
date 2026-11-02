"""Per-particle unknown retention, statistics, actual footprints and trajectory plots."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy import ndimage
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.surface_correspondence.common import *
from variants.surface_correspondence.model import particle_support,Config
CONFIG=Config(enabled=True)
PALETTE=np.array([[235,90,100],[40,210,180],[245,170,65],[175,130,245]])/255


def point_inside(projected,depth,area,mask):
    xy=np.rint(projected).astype(np.int64);good=np.isfinite(projected).all(-1)&(depth>.2)&np.isfinite(area)&(area>0)
    good&=(xy[:,0]>=0)&(xy[:,0]<400)&(xy[:,1]>=0)&(xy[:,1]<225)
    inside=good.copy();eroded=ndimage.binary_erosion(mask,iterations=1)
    inside[good]&=eroded[xy[good,1],xy[good,0]]
    return inside


def main():
    torch.set_num_threads(4);result=read(REPORT/'results.json');z=np.load(OUT/'surface_trajectories.npz')
    image=np.load(ROOT/'output/surface_support/image_tracks.npz');tracks=np.load(OUT/'fragment_tracks.npz')
    frames=z['frames'];ids=z['kernel_ids'];positions={int(k):i for i,k in enumerate(ids)}
    banks={b['name']:b for b in result['banks']};locals={};bank_ids={}
    for name,b in banks.items():
        with np.load(ROOT/b['bank_file']) as v:bank_ids[name]=v['kernel_ids'].copy()
        locals[name]=np.array([positions[int(k)] for k in bank_ids[name]])
    assigned=np.full((120,len(ids)),-1,dtype=np.int8);coverage=[];candidate_events=0
    for i,row in enumerate(result['frames']):
        proposals=[set() for _ in ids]
        for r in row['banks']:
            if not r['accepted']:continue
            candidate_events+=1;ix=locals[r['bank']];c=r['component']
            inside=point_inside(z['projected'][i,ix],z['depth'][i,ix],z['footprint_sqrt_det'][i,ix],image['components'][i]==c)
            good=particle_support(z['mass'][i,ix,c],z['mass'][i,ix,4],inside,z['raw_flow_error'][i,ix],CONFIG)
            for index in ix[good]:proposals[index].add(c)
        for k,values in enumerate(proposals):
            if len(values)==1:assigned[i,k]=next(iter(values))
        selected=np.where(assigned[i]>=0)[0]
        mass=float(z['mass'][i,selected,assigned[i,selected]].sum())
        coverage.append({'frame':int(frames[i]),'accepted_unique_particles':len(selected),'accepted_fragment_alpha_mass':mass,
          'visible_duck_alpha_mass':row['all_duck_alpha_mass'],'accepted_surface_alpha_coverage':mass/row['all_duck_alpha_mass'] if row['all_duck_alpha_mass']>0 else None,
          'unknown_particles_in_bank_union':int((assigned[i]<0).sum())})
    np.savez_compressed(OUT/'surface_assignments.npz',frames=frames,kernel_ids=ids,component_assignment=assigned)
    per_bank=[]
    for name,b in banks.items():
        future=[r for row in result['frames'] for r in row['banks'] if r['bank']==name and r['phase']=='forward_validation']
        visible=[r for r in future if r['component']>=0];accepted=[r for r in visible if r['accepted']]
        accepts=[row['frame'] for row in result['frames'] if any(r['bank']==name and r['accepted'] for r in row['banks'])]
        per_bank.append({'bank':name,'anchor':b['anchor'],'particle_count':b['particle_count'],'future_frames':len(future),
            'future_visible_reference_frames':len(visible),'accepted_future_frames':len(accepted),'accepted_frame_ids':accepts,
            'mean_accepted_fragment_alpha_coverage':float(np.mean([r['fragment_alpha_coverage'] for r in accepted])) if accepted else None,
            'mean_visible_fragment_alpha_coverage':float(np.mean([r['fragment_alpha_coverage'] for r in visible])) if visible else None})
    summary={'built_banks':len(banks),'rejected_banks':result['rejected_banks'],'unique_canonical_particles':len(ids),'bank_particle_counts_sum':sum(b['particle_count'] for b in banks.values()),
      'accepted_forward_bank_frame_events':candidate_events,'frames_with_some_confirmed_visible_surface':sum(c['accepted_unique_particles']>0 for c in coverage),
      'all120_mean_accepted_alpha_coverage':float(np.mean([c['accepted_surface_alpha_coverage'] for c in coverage])),
      'bank_rows':per_bank,'frames':coverage,'full_reconstruction_gate_passed':False,
      'gate_reason':'Only conditional local image-fragment / frozen Gaussian-row support established; unseen interaction identities and persistent complete objects unverified. No cross-anchor physical identity merging.',
      'no_new_Gaussian_training':True,'all_old_models_retained':True,'support_scores_are_not_physical_identity_probabilities':True}
    write(REPORT/'summary.json',summary)
    # Five-state bank timeline: prebirth, seed, accepted, visible failed, unknown.
    state=np.zeros((len(banks),120),int)
    for j,name in enumerate(banks):
        for i,row in enumerate(result['frames']):
            r=next(r for r in row['banks'] if r['bank']==name)
            state[j,i]=0 if r['phase']=='prebirth' else (1 if r['phase']=='seed' else (2 if r['accepted'] else (3 if r['component']>=0 else 4)))
    cmap=ListedColormap(['#ddd','#7ab8d5','#39a485','#d87a65','#c7b17c'])
    fig,(ax,ay)=plt.subplots(2,1,figsize=(14,7),gridspec_kw={'height_ratios':[3,1]},layout='constrained')
    im=ax.imshow(state,aspect='auto',interpolation='nearest',cmap=cmap,norm=BoundaryNorm(np.arange(-.5,5.5),5));ax.set_yticks(range(len(banks)),list(banks),fontsize=8)
    ticks=np.linspace(0,119,9).astype(int);ax.set_xticks(ticks,frames[ticks]);ax.set(title='Frozen canonical surface banks: full center AND full compositing-footprint gates',xlabel='original training frame')
    cb=fig.colorbar(im,ax=ax,ticks=range(5),pad=.01);cb.ax.set_yticklabels(['prebirth','seed, not validation','accepted local surface','visible, joint gate failed','image identity unknown'])
    ay.plot(frames,[c['accepted_surface_alpha_coverage']*100 for c in coverage]);ay.set(ylabel='accepted alpha coverage (%)',xlabel='original training frame',ylim=(0,100),title='Unique current particles only; unresolved assignments excluded; no double counting across anchors')
    fig.savefig(OUT/'support_timeline.png',dpi=140);plt.close(fig)
    # A fixed lowest-ID point from each anchor's first successful bank, not the
    # best-looking trajectory. This is model motion, never physical3D GT.
    chosen=[]
    for anchor in [2,63,86,134,172]:
        candidates=[name for name,b in banks.items() if b['anchor']==anchor]
        if candidates:chosen.append(candidates[0])
    fig=plt.figure(figsize=(14,7));grid=fig.add_gridspec(2,2,width_ratios=[1,1.15],wspace=.3,hspace=.4)
    ax=fig.add_subplot(grid[:,0],projection='3d');ay=fig.add_subplot(grid[0,1]);az=fig.add_subplot(grid[1,1])
    fig.subplots_adjust(left=.025,right=.98,bottom=.09,top=.94)
    for j,name in enumerate(chosen):
        local=locals[name][0];p=z['centers'][:,local];a=np.array([any(r['bank']==name and r['accepted'] for r in row['banks']) for row in result['frames']])
        a&=assigned[:,local]>=0
        color=plt.get_cmap('tab10')(j);ax.plot(*p.T,color=color,alpha=.6,label=f'{name} row{ids[local]}');ax.scatter(*p[a].T,color=color,s=8)
        ay.plot(frames,z['footprint_sqrt_det'][:,local],color=color,label=name)
        az.plot(frames,z['mass'][:,local,:4].sum(-1),color=color,label=name)
    ax.set(title='Same canonical row, full3D center path',xlabel='world x',ylabel='world y',zlabel='world z');ax.legend(fontsize=6)
    ay.set(title='Actual projected covariance size',xlabel='original frame',ylabel='sqrt(det covariance), px²',yscale='log')
    az.set(title='Full visible footprint on yellow regions',xlabel='original frame',ylabel='integrated alpha mass, pixels');az.legend(fontsize=7)
    fig.savefig(OUT/'center_footprint_3d.png',dpi=145);plt.close(fig)
    # Actual CUDA compositor visuals, preserving every Gaussian and opacity.
    g,d,proxy=setup();metadata=read(TRAIN/'transforms_train.json');lookup={int(Path(f['file_path']).stem):f for f in metadata['frames']}
    selected=sorted(set([2,19,43,63,74,86,108,134,172,200]+[b['seed_frames'][-1] for b in banks.values()]+[int(frames[b['seed_end_index']+1]) for b in banks.values()]))
    folder=OUT/'review';folder.mkdir(exist_ok=True);visual_checks=[]
    early_color=torch.zeros(len(g.get_xyz),3,device='cuda')
    early_ids=[];early_labels=[]
    for name,b in banks.items():
        if b['anchor']==2:
            early_color[torch.as_tensor(bank_ids[name],device='cuda')]=torch.as_tensor(PALETTE[b['local_seed']],device='cuda',dtype=torch.float32)
            early_ids.extend(bank_ids[name].tolist());early_labels.extend([b['local_seed']]*len(bank_ids[name]))
    early_local=np.array([positions[k] for k in early_ids]);early_labels=np.array(early_labels)
    for number in selected:
        i=int(np.where(frames==number)[0][0]);cam,w,_,_,_,_=geometry(g,d,lookup[number],metadata)
        with Image.open(TRAIN/(lookup[number]['file_path']+'.png')) as im:rgba=np.asarray(im.convert('RGBA').resize((400,225),Image.Resampling.BILINEAR))/255
        gt=rgba[...,:3]*rgba[...,3:4];image_labels=np.full((225,400),-1,int)
        active_anchor=max(a for a,b in enumerate(read(REPORT/'image_fragments.json')) if b['start_index']<=i)
        for k,c in enumerate(tracks['component_mapping'][active_anchor,i]):
            if c>=0:image_labels[image['components'][i]==c]=k
        visible=np.repeat(gt.mean(-1,keepdims=True),3,-1)*.5
        for k in range(4):visible[image_labels==k]=.3*gt[image_labels==k]+.7*PALETTE[k]
        color=torch.full((len(g.get_xyz),3),.12,device='cuda');good=np.where(assigned[i]>=0)[0]
        color[torch.as_tensor(ids[good],device='cuda')]=torch.as_tensor(PALETTE[assigned[i,good]],device='cuda',dtype=torch.float32)
        with torch.no_grad():
            p=render(proxy,cam,w,color);colored=p['render'].permute(1,2,0).cpu().numpy();alpha=p['alpha'].squeeze().cpu().numpy()
            e=render(proxy,cam,w,early_color)['render'].permute(1,2,0).cpu().numpy()
        expected=np.ones(3)*.12*alpha.sum()
        for k in range(4):
            active=np.where(assigned[i]==k)[0];expected+=(PALETTE[k]-.12)*z['mass'][i,active,4].sum()
        numeric=float(np.max(np.abs(colored.sum((0,1))-expected)/np.maximum(np.abs(expected),1)));assert numeric<.002
        expected_early=np.sum(z['mass'][i,early_local,4,None]*PALETTE[early_labels],axis=0)
        early_error=float(np.max(np.abs(e.sum((0,1))-expected_early)/np.maximum(np.abs(expected_early),1)));assert early_error<.002
        overlays=[]
        for subset,labels in [(early_local,early_labels),(good,assigned[i,good])]:
            canvas=Image.fromarray((gt*255).round().astype('uint8'));draw=ImageDraw.Draw(canvas)
            for xy,k in zip(z['projected'][i,subset],labels):
                x,y=xy
                if np.isfinite(xy).all() and 0<=x<400 and 0<=y<225:draw.ellipse((x-.6,y-.6,x+.6,y+.6),fill=tuple((PALETTE[k]*255).astype(int)))
            overlays.append(np.asarray(canvas)/255)
        values=[gt,visible,overlays[0],e,overlays[1],colored]
        titles=[f'train GT {number}',f'local fragments anchor{[2,63,86,134,172][active_anchor]}','frozen early bank centers','same early banks: full footprint','confirmed current surface centers','confirmed full footprint; grey unknown']
        canvas=Image.new('RGB',(2400,265),'#182a3d');draw=ImageDraw.Draw(canvas)
        for j,(value,title) in enumerate(zip(values,titles)):
            canvas.paste(Image.fromarray((np.clip(value,0,1)*255).round().astype('uint8')),(400*j,40));draw.text((400*j+4,10),title,fill='white')
        canvas.save(folder/f'frame{number:04d}.png');visual_checks.append({'frame':number,'current_surface_integral_relative_error':numeric,'early_cohort_integral_relative_error':early_error})
    write(REPORT/'visual_contracts.json',{'passed':True,'full_compositor_visuals':visual_checks,'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}})
    html='''<!doctype html><meta charset="utf-8"><title>Visible image fragments and frozen3D surfaces</title><style>body{background:#142538;color:#eee;font:16px system-ui;margin:24px}img{width:100%;margin:12px 0}a{color:#8ee}input{width:65%}</style><h1>Image-fragment / canonical Gaussian surface correspondence</h1><p>Same frozen rows are tested over all120 training frames, with full center motion and actual RGB alpha contributions. Local visible support is conditional evidence, NOT physical object/material identity. Grey means unknown; independent anchors are not merged. Three seed frames select each bank once; later frames validate it.</p><a href="../../experiments/surface_correspondence/assessment.md">Assessment</a> | <a href="../../experiments/surface_correspondence/summary.json">Results</a> | <a href="../../experiments/surface_correspondence/independent_verification.json">Independent verification</a><img src="support_timeline.png"><img src="center_footprint_3d.png"><p><input id="s" type="range" min="0" max="MAX" value="0"><b id="label"></b></p><img id="frame"><p>GT | latest local fragment evidence | frozen early centers | full early footprint | confirmed current centers | full current footprint / unknown grey. Colors are local component labels, not cross-time physical IDs.</p><script>const fs=FRAMES;const s=document.querySelector('#s');function show(){const f=fs[+s.value];document.querySelector('#label').textContent='Original training frame '+f;document.querySelector('#frame').src='review/frame'+String(f).padStart(4,'0')+'.png'}s.oninput=show;show()</script>'''.replace('MAX',str(len(selected)-1)).replace('FRAMES',json.dumps(selected))
    (OUT/'index.html').write_text(html,encoding='utf-8');print('SURFACE FIGURES COMPLETE',summary['built_banks'],summary['accepted_forward_bank_frame_events'],summary['all120_mean_accepted_alpha_coverage'],flush=True)


if __name__=='__main__':main()
