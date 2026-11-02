"""Read-only same-material-anchor probes; no new training or model selection."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
sys.path.insert(0,str(ROOT/'variants/momentum_guard/code'))
from scene_PIDG.gaussian_model import GaussianModel
from scene_PIDG.deform_model import DeformModel
from variants.momentum_guard.run import assert_frozen,REPORTS
from variants.flow_comparison.temporal import position,derivative
from variants.flow_comparison.local_momentum import Graph,residual
from variants.unified_flow.model import predict_stress
spec=importlib.util.spec_from_file_location('fixed_probe_motion',ROOT/'output/flow_comparison/evaluate_motion_v2.py')
motion=importlib.util.module_from_spec(spec);spec.loader.exec_module(motion)
torch.set_num_threads(4);assert_frozen(ROOT/'variants/momentum_guard/code')
prefix=ROOT/'runs/pidg_unified_flow/shared_prefix/rubber_duck/checkpoints/resume.pt'
state=torch.load(prefix,map_location='cpu',weights_only=False)
x=state['gaussian_parameters']['_xyz'].cuda();del state
graph=Graph();all_ids,first,anchors,n_first,n_anchor=graph.sample(x,50000,count=64,k=12)
query=x[all_ids];rms=lambda a:float(a.double().square().mean().sqrt())
records=[]
for directory in [ROOT/'runs/pidg_unified_flow/L1_shared_stress/rubber_duck',ROOT/'runs/pidg_momentum_guard/M1_guarded_shared/rubber_duck']:
    os.environ['PIDG_METRIC_CONFIG']=str(directory/'metric_config.json')
    g,d=motion.load(directory,'best')
    with torch.no_grad():
        for frame in [2,83,155]:
            t=(frame-1)/199;f=lambda s:position(d,query,s)
            now=f(t);v=derivative(f,t,.001);stress=predict_stress(d,query,t)
            accel=derivative(f,t,1/199,2)
            r,valid=residual(now,v,stress,accel,g.force_sigma,all_ids,first,anchors,n_first,n_anchor)
            ar=accel[torch.searchsorted(all_ids,anchors)]
            pred_only,pvalid=residual(now,v,stress,accel,0.,all_ids,first,anchors,n_first,n_anchor)
            rate_only,rvalid=residual(now,v,torch.zeros_like(stress),accel,g.force_sigma,all_ids,first,anchors,n_first,n_anchor)
            assert torch.equal(valid,pvalid) and torch.equal(valid,rvalid)
            row={'model':directory.parent.name,'frame':frame,'snapshot':'best','canonical_query_count':len(query),
                'valid_fraction':float(valid.float().mean()),'raw_huber':float(torch.nn.functional.huber_loss(r[valid],torch.zeros_like(r[valid]))),
                'residual_rms':rms(r[valid]),'acceleration_rms_on_valid_anchors':rms(ar[valid]),
                'divergence_rms_on_valid_anchors':rms((ar-r)[valid]),'stress6_rms':rms(stress),'force_sigma':float(g.force_sigma.detach()),
                'predicted_stress_divergence_rms':rms((ar-pred_only)[valid]),
                'rate_correction_divergence_rms':rms((ar-rate_only)[valid]),
                'residual_rms_without_rate_correction':rms(pred_only[valid])}
            row['acceleration_step_sweep']=[]
            for factor in [2,1,.5,.25,.125,.0625]:
                a=derivative(f,t,1/199*factor,2)
                row['acceleration_step_sweep'].append({'factor':factor,'h':1/199*factor,
                    'rms':rms(a),'relative_difference_to_base':rms(a-accel)/max(rms(accel),1e-12)})
            row['velocity_step_sweep']=[]
            for factor in [.5,1,2]:
                vv=derivative(f,t,.001*factor)
                row['velocity_step_sweep'].append({'factor':factor,'rms':rms(vv),
                    'relative_difference_to_base':rms(vv-v)/max(rms(v),1e-12)})
            records.append(row)
    del g,d;torch.cuda.empty_cache()
assert_frozen(ROOT/'variants/momentum_guard/code')
result={'status':'complete','probes':records,
    'purpose':'Same 3k canonical points, same 64 anchors/neighborhood indices and training times; deterministic continuous fields evaluated at best snapshots, independent of final Gaussian pruning. No optimizer or checkpoint changes.',
    'limitations':'Not physical GT or an exact continuum derivative. Prefix points may have been pruned later. Neighborhood validity can differ under deformation. Shrinking FP32 finite differences also expose cancellation; no guarantee of an exact finest-step reference.'}
(REPORTS/'fixed_field_probes.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result),flush=True)
