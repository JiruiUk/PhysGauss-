"""Observed train RGB and fixed diagnostic panels only; no new rendered views."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay,QhullError,ConvexHull
from variants.staged_reconstruction.i10_joint_scale import io


def hull_area(points):
    if len(points)<3:return 0.
    try:return float(ConvexHull(points).volume/90000)
    except QhullError:return 0.


def run():
    a=io.read(io.REPORT/'audit_v2.json');old=io.read(io.ROOT/io.read(io.BASE/'protocol.json')['old_audit'])
    p=io.read(io.BASE/'protocol.json');io.OUT.mkdir(parents=True,exist_ok=True);files=[]
    for i in p['fixed_review_frames']:
        with np.load(io.ROOT/p['raw_maps']/f'{i:03d}.npz') as f:raw=f['relative']
        rgb=io.rgb(i);cal=a['candidate']['calibration'][i];bank=a['anchor_bank'][i]
        fig,axes=plt.subplots(2,3,figsize=(14,5.8));fig.patch.set_facecolor('#171c24')
        for ax in axes.flat:ax.set_facecolor('black');ax.axis('off')
        axes[0,0].imshow(rgb);axes[0,0].set_title('Observed train RGB; no geometry GT',color='white',fontsize=10)
        axes[0,1].imshow(rgb)
        for color,split in [('#1de5ff','fit0'),('#ffb12b','fit1'),('#ff55dc','check')]:
            uv=np.array([q['uv'] for q in bank if q['split']==split]).reshape(-1,2)
            if len(uv):axes[0,1].scatter(uv[:,0],uv[:,1],s=14,facecolors='none',edgecolors=color,label=split)
        axes[0,1].set_title('Expanded deduplicated anchors; check groups never fit',color='white',fontsize=10)
        axes[0,1].legend(fontsize=7,loc='lower left')
        axes[0,2].imshow(raw,cmap='viridis');axes[0,2].set_title('Frozen Small relative inverse depth; display scale',color='white',fontsize=10)
        axes[1,0].imshow(rgb)
        oc=old['calibration'][i];uv=np.array(oc['uv']).reshape(-1,2)
        if len(uv):axes[1,0].scatter(uv[:,0],uv[:,1],s=12,facecolors='none',edgecolors='#bfc4cf')
        axes[1,0].set_title(f'Original L7-P anchors; calibrated={oc["passed"]}',color='white',fontsize=10)
        depth=np.full(raw.shape,np.nan)
        if cal['passed']:
            uv=np.array(cal['uv'])[cal['fit_ids']];y,x=np.mgrid[:225,:400]
            inside=Delaunay(uv).find_simplex(np.c_[x.reshape(-1),y.reshape(-1)])>=0
            inv=raw.astype(float)*cal['coefficients'][0]+cal['coefficients'][1]
            valid=inside.reshape(raw.shape)&(inv>0);depth[valid]=1/inv[valid]
        axes[1,1].imshow(depth,cmap='viridis');axes[1,1].set_title(f'Joint depth ONLY in current fit hull; passed={cal["passed"]}',color='white',fontsize=10)
        axes[1,2].imshow(rgb);target=a['candidate']['targets'].get(str(i),[])
        for q in target:axes[1,2].scatter(*q['uv'],c='#00ff55',marker='+',s=100)
        axes[1,2].set_title(f'Admissible fit observations={len(target)}; global gate=False',color='white',fontsize=10)
        fig.suptitle(f'L7-J train index {i}; {cal["reason"]}; black/blank=unknown, not zero error',color='white',fontsize=11)
        fig.tight_layout();path=io.OUT/f'joint_scale_{i:03d}.png';fig.savefig(path,dpi=125,facecolor=fig.get_facecolor());plt.close(fig);files.append(path)
    fig,axes=plt.subplots(3,1,figsize=(13,8));cases=[old['calibration'],a['framewise_control']['calibration'],a['candidate']['calibration']]
    matrix=np.array([[c['passed'] for c in row] for row in cases],float)
    axes[0].imshow(matrix,aspect='auto',cmap='Greens',vmin=0,vmax=1,extent=(-.5,119.5,2.5,-.5))
    axes[0].set_yticks([0,1,2],['Original L7-P:24','Expanded framewise:26','Expanded joint:39']);axes[0].set_title('Scale checks per observed training frame; white=not admitted')
    oldh=[hull_area(np.array(c['uv'])[c['fit_ids']]) for c in old['calibration']]
    newh=[c['coverage']['hull_area_fraction'] for c in a['candidate']['calibration']]
    axes[1].plot(oldh,label='Original fit hull');axes[1].plot(newh,label='Expanded group-held fit hull');axes[1].legend();axes[1].set_ylabel('Image area fraction');axes[1].set_title('Potential spatial support, including failed calibrations; NOT GT surface coverage')
    motion=[q for q in a['candidate']['motion'] if 'row' in q];motion.sort(key=lambda q:(q['start'],q['id']))
    available=np.array([q['local_eligible'] for q in motion]);axes[2].imshow(available,aspect='auto',cmap='Greens',vmin=0,vmax=1)
    axes[2].set_title('122 original native-eligible track hypotheses; green=local depth usable, 1 track passes motion check')
    axes[2].set_xlabel('Observation within each12-frame track (even=fit, odd=check)');axes[2].set_ylabel('Tracks ordered by start')
    for ax in axes[:2]:
        for t in [39.5,79.5]:ax.axvline(t,color='gray',linestyle='--',linewidth=.7)
    fig.tight_layout();path=io.OUT/'temporal_coverage.png';fig.savefig(path,dpi=125);plt.close(fig);files.append(path)
    passing=[q for q in motion if q['passed']]
    if passing:
        q=sorted(passing,key=lambda q:q['id'])[0];fig,axes=plt.subplots(3,4,figsize=(13,6))
        for t,ax in enumerate(axes.flat):
            ax.imshow(io.rgb(q['start']+t));ax.scatter(*q['uv'][t],marker='+',color='#00ff55',s=100);ax.axis('off')
            ax.set_title(f'train {q["start"]+t}; {"fit" if t%2==0 else "check"}',fontsize=9)
        fig.suptitle(f'POSTHOC illustration only: smallest passing ID {q["id"]}, row {q["row"]}; parent-conditioned motion, not GT')
        fig.tight_layout();path=io.OUT/'local_track_example.png';fig.savefig(path,dpi=125);plt.close(fig);files.append(path)
    io.write(io.REPORT/'figures.json',dict(files={p.relative_to(io.ROOT).as_posix():io.sha(p) for p in files},
        fixed_frames=p['fixed_review_frames'],posthoc_track_selection='smallest passing ID, illustration only; no input/gate changes',
        new_training_results=False,source_sha256=io.sha(Path(__file__)),
        all120_potential_hull_mean=dict(original=float(np.mean(oldh)),expanded=float(np.mean(newh)))))
    print('JOINT_SCALE_FIGURES',len(files),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-joint-scale-audit',action='store_true');args=p.parse_args()
    if not args.enable_joint_scale_audit:raise SystemExit('Joint scale audit default off')
    with io.lock():run()


if __name__=='__main__':main()
