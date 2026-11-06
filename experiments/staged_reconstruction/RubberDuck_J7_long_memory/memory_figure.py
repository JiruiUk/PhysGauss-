"""Descriptive observed-memory support; no training input or threshold selection."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from variants.staged_reconstruction.prepare_data import ROOT,read,sha,write

B=Path(__file__).resolve().parent
O=ROOT/'output/staged_reconstruction/RubberDuck_J7_long_memory/v1'

def main():
    bank=read(B/'v1/bank.json');health=read(B/'v1/initial_health.json')
    memories=[m for m in bank['memories'] if m['binding']['accepted']]
    states=health['controller']['memories'];O.mkdir(parents=True,exist_ok=True)
    fig,(ax,bx)=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[3,1]},layout='constrained')
    counts=np.zeros(120,int)
    for index,(m,s) in enumerate(zip(memories,states)):
        color='#168a66' if s['status']=='active' else '#a8afb5'
        ax.plot(m['frames'],[index]*len(m['frames']),'-o',lw=2,ms=3,color=color)
        if s['status']=='active':counts[m['frames']]+=1
    ax.set(xlim=(-1,120),ylabel='J7-revalidated bound memory',title='Observed temporal memory: 17 active / 21 geometrically bound (641 retained)')
    ax.set_yticks(range(len(memories)));ax.set_yticklabels([str(m['binding']['row']) for m in memories],fontsize=7)
    ax.grid(axis='x',alpha=.2)
    bx.bar(np.arange(120),counts,width=1,color='#168a66')
    bx.set(xlim=(-1,120),xlabel='Training image / exact time index',ylabel='Active memories',title='Initial observed support; no invented observations in gaps')
    for axis in [ax,bx]:
        axis.axvline(39.5,color='#888',ls='--',lw=1);axis.axvline(79.5,color='#888',ls='--',lw=1)
    fig.suptitle('J7 + explicit observation memory (J8): coverage limitation',fontsize=14)
    path=O/'memory_coverage.png';fig.savefig(path,dpi=160);plt.close(fig)
    write(B/'v1/memory_figure.json',dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),
        active_observations=int(counts.sum()),frame_thirds=[int(counts[i:i+40].sum()) for i in [0,40,80]],
        descriptive_only=True,independent_geometry_GT=False,source_sha256=sha(Path(__file__))))

if __name__=='__main__':main()
