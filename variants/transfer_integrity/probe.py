"""Real full-C1 gradients and isolated update; original state is never written."""
import os
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from variants.transfer_integrity.run import ROOT,RUNS,REPORT,SOURCE,DATA,config
from variants.transfer_integrity.runtime import Tuning
from variants.transfer_integrity.data import prepare
from variants.association_training.evaluate import load,render_call
from variants.shape_coherence.observations import camera
from variants.flow_comparison.temporal import warp_at
from variants.flow_comparison.flow import projected
from variants.integrity_tuning.model import coverage_objective
from variants.coverage_tuning.data import read_alpha
from variants.transfer_integrity.model import project
from variants.metric_tuning.runtime import write_json,read_json,sha256


def main():
    torch.set_num_threads(4);binding=prepare();folder=RUNS/'_probe';folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'metric_config.json',config(True,binding));os.environ['PIDG_METRIC_CONFIG']=str(folder/'metric_config.json')
    protected=sha256(SOURCE/'checkpoints/resume.pt');g,d,proof=load(SOURCE)
    for name in ['_xyz','_opacity','_scaling','_rotation','_features_dc','_features_rest']:getattr(g,name).requires_grad_(True)
    g.velocity_net.requires_grad_(True);d.grid4d.requires_grad_(True);d.deform.requires_grad_(True)
    t=Tuning(folder/'metric_config.json',folder);t.transfer_deform=d;t.initialize_data(DATA)
    attention=d.deform.attention_score
    loss,terms=t.geometry_loss(50010,g)
    assert d.deform.attention_score is attention
    meta=read_json(DATA/'transforms_train.json');frame=next(f for f in meta['frames'] if f['file_path'].endswith('0086'))
    cam=camera(frame,meta,1600);w=warp_at(d,g.get_xyz.detach(),85/199);package=render_call(cam,g,w)
    target=read_alpha(DATA/(frame['file_path']+'.png'),(cam.image_width,cam.image_height)).to(package['alpha'])
    coverage=coverage_objective(package['alpha'],target)
    total=loss+.002*coverage
    x,_=t.at(g,torch.arange(128,device='cuda'),85/199)
    uv,_=project(x,*t.transfer_cameras[86]);expected,_=projected(x,g.get_scaling[:128]+w['d_scaling'][:128],g.get_rotation[:128]+w['d_rotation'][:128],camera(frame,meta,400))
    assert (uv-expected).abs().max()<.003
    groups={'grid':list(d.grid4d.parameters()),'deformation':list(d.deform.parameters()),'canonical_centers':[g._xyz],
            'opacity':[g._opacity],'scales_rotations':[g._scaling,g._rotation],'velocity_stress':list(g.velocity_net.parameters())}
    gradients={}
    for key,params in groups.items():
        values=torch.autograd.grad(total,params,retain_graph=True,allow_unused=True)
        assert all(v.isfinite().all() for v in values if v is not None)
        gradients[key]=sum(float(v.abs().sum()) for v in values if v is not None)
    assert all(gradients[k]>0 for k in ['grid','deformation','canonical_centers','opacity','scales_rotations'])
    assert gradients['velocity_stress']==0
    params=[p for k,group in groups.items() if k!='velocity_stress' for p in group]
    before=[p.detach().clone() for p in params];optimizer=torch.optim.Adam(params,lr=1e-6)
    optimizer.zero_grad();total.backward();optimizer.step()
    change=sum(float((p.detach()-a).abs().sum()) for p,a in zip(params,before))
    assert change>0 and sha256(SOURCE/'checkpoints/resume.pt')==protected
    result={'status':'passed','full_source_state':proof,'native_alpha_renderer_width':1600,
            'loss':float(total.detach()),'terms':{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in terms.items()},
            'finite_gradient_L1':gradients,'actual_parameter_L1_change':change,'historical_checkpoint_unchanged':True,
            'camera_projection_max_error_400px':float((uv-expected).abs().max().detach()),'attention_cache_restored':True}
    write_json(REPORT/'optimizer_probe.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
