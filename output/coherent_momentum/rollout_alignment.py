"""Read-only matched-material 3D force rollout versus rendered trajectory."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
from variants.coherent_momentum.run import RUNS,NAME,REPORTS,assert_frozen
from variants.coherent_momentum.evaluate import motion_module,CONTROL
from variants.coherent_momentum.operator import kinematics,momentum
from variants.flow_comparison.local_momentum import Graph
from variants.flow_comparison.temporal import position
from variants.unified_flow.model import predict_stress
assert_frozen(ROOT/'variants/coherent_momentum/code');module=motion_module()
state=torch.load(ROOT/'runs/pidg_unified_flow/shared_prefix/rubber_duck/checkpoints/resume.pt',map_location='cpu',weights_only=False)
canonical=state['gaussian_parameters']['_xyz'].cuda();del state
_,first,anchors,_,neighbor=Graph().sample(canonical,50000,count=64,k=32)
query=canonical[first];local=torch.searchsorted(first,anchors);ring=torch.searchsorted(first,neighbor)
records=[];selection=None;h=1/199;t=82/199
for directory in [CONTROL,RUNS/NAME/'rubber_duck']:
    os.environ['PIDG_METRIC_CONFIG']=str(directory/'metric_config.json')
    g,d=module.load(directory,'best')
    with torch.no_grad():
        x,v,a,xm,xp,times=kinematics(lambda s:position(d,query,s),t,h)
        stress=predict_stress(d,query,times[1]).double()
        r,valid,parts=momentum(x,v,stress,a,g.force_sigma,local,ring)
        rollout=2*x[local]-xm[local]+h*h*(a[local]-r)
        records.append({'model':directory.parent.name,'x_minus':xm[local].cpu().tolist(),
            'x_now':x[local].cpu().tolist(),'x_render_next':xp[local].cpu().tolist(),
            'x_physics_next':rollout.cpu().tolist(),'velocity_norm':v[local].norm(dim=-1).cpu().tolist(),
            'valid':valid.cpu().tolist(),'residual_rms':float(r[valid].square().mean().sqrt())})
    del g,d;torch.cuda.empty_cache()
common=np.array(records[0]['valid'])&np.array(records[1]['valid'])
available=np.flatnonzero(common)
selection=available[np.argsort(np.array(records[0]['velocity_norm'])[available])[-8:]]
fig=plt.figure(figsize=(13,6),layout='constrained');allpoints=[];axes=[]
for panel,row in enumerate(records):
    ax=fig.add_subplot(1,2,panel+1,projection='3d');axes.append(ax)
    xm,x,xp,phys=[np.asarray(row[k])[selection] for k in ['x_minus','x_now','x_render_next','x_physics_next']]
    for j in range(len(selection)):
        line=np.stack([xm[j],x[j],xp[j]])
        ax.plot(*line.T,color='#246bc3',linewidth=2,label='Rendered trajectory' if j==0 else None)
        ax.plot(*np.stack([x[j],phys[j]]).T,color='#e46e31',linewidth=1.7,linestyle='--',label='Stress-driven rollout' if j==0 else None)
        ax.plot(*np.stack([xp[j],phys[j]]).T,color='#999999',linewidth=1)
    ax.scatter(*xp.T,color='#246bc3',s=20);ax.scatter(*phys.T,color='#e46e31',s=20,marker='x')
    ax.set(title=row['model'],xlabel='World x',ylabel='World y',zlabel='World z');ax.legend(fontsize=8)
    allpoints.extend([xm,x,xp,phys])
points=np.concatenate(allpoints);center=.5*(points.min(0)+points.max(0));radius=max(np.ptp(points,axis=0).max()*.55,.001)
for ax in axes:
    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
    ax.set_box_aspect((1,1,1))
fig.suptitle('Frame 83: same material points and coherent operator; predicted 3D fields, no physical GT\n8 common valid anchors with largest reference speed; equal axes in both panels',fontsize=11)
out=Path(__file__).parent;fig.savefig(out/'rollout_alignment_3d.png',dpi=190);plt.close(fig)
result={'frame':83,'h':h,'canonical_ids':anchors.cpu().tolist(),'selected_anchor_indices':selection.tolist(),
    'selection_rule':'8 common valid anchors with largest M1 speed, fixed for both models',
    'records':records,'not_independent_validation':'rollout gap equals h^2 times acceleration residual; this is visualization of the same residual, not physical ground truth'}
(REPORTS/'rollout_alignment_3d.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print('3D alignment figure and vector data saved')
