"""Actual native coverage + new geometry gradients on the fresh common state."""
from pathlib import Path
import os
import json
import sys
import torch
from .prefix import ROOT,REPORT,RUNS,DATA
sys.path.insert(0,str(ROOT/'variants/scratch_integrity/code'))
from .runtime import Tuning
from .run import configuration
from variants.association_training.evaluate import load,render_call
from variants.shape_coherence.observations import camera
from variants.flow_comparison.temporal import warp_at
from variants.coverage_tuning.data import read_alpha
from variants.integrity_tuning.model import coverage_objective
from variants.metric_tuning.runtime import read_json,write_json,sha256


def main():
    torch.set_num_threads(4);b=read_json(REPORT/'data_binding.json');source=Path(b['source_directory'])
    folder=RUNS/'_probe';folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'metric_config.json',configuration(True,b));os.environ['PIDG_METRIC_CONFIG']=str(folder/'metric_config.json')
    g,d,proof=load(source)
    for key in ['_xyz','_opacity','_scaling','_rotation']:getattr(g,key).requires_grad_(True)
    g.velocity_net.requires_grad_(True);d.grid4d.requires_grad_(True);d.deform.requires_grad_(True)
    t=Tuning(folder/'metric_config.json',folder);t.transfer_deform=d;t.initialize_data(DATA)
    old=d.deform.attention_score;loss,terms=t.geometry_loss(b['source_iteration']+20,g)
    assert old is d.deform.attention_score
    meta=read_json(DATA/'transforms_train.json');f=next(f for f in meta['frames'] if f['file_path'].endswith('0086'))
    cam=camera(f,meta,1600);w=warp_at(d,g.get_xyz.detach(),85/199);package=render_call(cam,g,w)
    target=read_alpha(DATA/(f['file_path']+'.png'),(1600,900)).to(package['alpha'])
    total=loss+.002*coverage_objective(package['alpha'],target)
    groups={'grid':list(d.grid4d.parameters()),'deformation':list(d.deform.parameters()),'canonical_centers':[g._xyz],
            'opacity':[g._opacity],'scales_rotations':[g._scaling,g._rotation],'velocity_stress':list(g.velocity_net.parameters())}
    gradients={}
    for key,params in groups.items():
        grads=torch.autograd.grad(total,params,allow_unused=True,retain_graph=True)
        assert all(v.isfinite().all() for v in grads if v is not None)
        gradients[key]=sum(float(v.abs().sum()) for v in grads if v is not None)
    assert all(gradients[k]>0 for k in groups if k!='velocity_stress') and gradients['velocity_stress']==0
    params=[p for k,ps in groups.items() if k!='velocity_stress' for p in ps];before=[p.detach().clone() for p in params]
    optimizer=torch.optim.Adam(params,lr=1e-6);optimizer.zero_grad();total.backward();optimizer.step()
    delta=sum(float((p.detach()-q).abs().sum()) for p,q in zip(params,before))
    assert delta>0 and sha256(source/'checkpoints/resume.pt')==b['source_checkpoint_sha256']
    write_json(REPORT/'optimizer_probe.json',{'status':'passed','fresh_source_state':proof,'attention_cache_restored':True,
               'finite_gradient_L1':gradients,'actual_parameter_L1_change':delta,'native_renderer_width':1600,
               'terms':{k:float(v) if torch.is_tensor(v) else v for k,v in terms.items()},'original_source_unchanged':True})


if __name__=='__main__':main()
