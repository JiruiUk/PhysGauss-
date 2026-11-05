"""Read-only visualization of actual persistent Gaussian 3D trajectories."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from variants.staged_reconstruction.i10_track_train import runtime as a
from variants.staged_reconstruction.i10_tracking.io import ROOT,DATA,read,write,sha
from experiments.staged_reconstruction.RubberDuck_I10_tracking.training.evaluate import load_models


@torch.no_grad()
def run():
    a.install(True);a.r.verify();torch.set_num_threads(4)
    _,models,_=load_models();p=read(a.BASE/'protocol.json');evidence=read(ROOT/p['evidence'])
    frames=read(DATA/'transforms_train.json')['frames'];fig=plt.figure(figsize=(15,5));values={};groups=[]
    for col,start in enumerate([0,52,108]):
        tracks=[r for r in evidence['rows'] if r['start']==start and r.get('binding',{}).get('accepted')]
        ids=sorted({r['binding']['row'] for r in tracks});assert ids
        ax=fig.add_subplot(1,3,col+1,projection='3d');colormap=plt.get_cmap('tab20')
        for label,scene in models.items():
            coords=np.stack([scene.dynamic_at(frames[i]['time']).get_xyz[ids].cpu().numpy() for i in range(start,start+12)])
            values[f'{start}_{label}']=coords
            if label=='rgb_only':continue
            for j,point_id in enumerate(ids):
                ax.plot(*coords[:,j].T,linestyle='--' if label=='J6_parent' else '-',
                    color='#888888' if label=='J6_parent' else colormap(j%20),alpha=.5 if label=='J6_parent' else .9,linewidth=1)
        groups.append(dict(start=start,Gaussian_rows=ids,training_frame_indices=list(range(start,start+12))))
        ax.set_title(f'Train indices {start}..{start+11}: {len(ids)} point IDs')
        ax.set_xlabel('world x');ax.set_ylabel('world y');ax.set_zlabel('world z');ax.view_init(elev=20,azim=-55)
    fig.suptitle('Persistent 3D Gaussian paths: dashed gray=J6 parent; solid=RGB+tracks\nModel-inferred depth; colors identify point rows, not physical objects or 3D ground truth',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.90]);path=a.OUT/'persistent_3D_trajectories.png';fig.savefig(path,dpi=160);plt.close(fig)
    archive=a.RUNS/'persistent_3D_trajectories.npz';np.savez_compressed(archive,**values)
    write(a.REPORT/'trajectory_visual.json',dict(groups=groups,image=path.relative_to(ROOT).as_posix(),image_sha256=sha(path),
        arrays=archive.relative_to(ROOT).as_posix(),arrays_sha256=sha(archive),source_sha256=sha(Path(__file__)),
        all_three_models_saved=True,independent_3D_GT=False,physical_object_count_or_identity=False,optimizer_updates=0))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--enable-trajectory-view',action='store_true')
    if not parser.parse_args().enable_trajectory_view:raise SystemExit('Trajectory visualization default off')
    run()
