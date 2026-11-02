"""Read-only shared-anchor decomposition and h/k sensitivity after selection."""
import json
import os
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
from variants.coherent_momentum.run import ROOT,RUNS,NAME,REPORTS,assert_frozen
from variants.coherent_momentum.evaluate import motion_module,CONTROL
from variants.coherent_momentum.operator import kinematics,momentum
from variants.flow_comparison.local_momentum import Graph
from variants.flow_comparison.temporal import position
from variants.unified_flow.model import predict_stress
assert_frozen(ROOT/'variants/coherent_momentum/code');module=motion_module()
state=torch.load(ROOT/'runs/pidg_unified_flow/shared_prefix/rubber_duck/checkpoints/resume.pt',map_location='cpu',weights_only=False)
canonical=state['gaussian_parameters']['_xyz'].cuda();del state
_,first,anchors,_,neighbor=Graph().sample(canonical,50000,count=64,k=48)
query=canonical[first];local=torch.searchsorted(first,anchors);rms=lambda z:float(z.double().square().mean().sqrt())
records=[]
for directory in [CONTROL,RUNS/NAME/'rubber_duck']:
    os.environ['PIDG_METRIC_CONFIG']=str(directory/'metric_config.json');g,d=module.load(directory,'best')
    with torch.no_grad():
        for frame in [2,83,155]:
            t=(frame-1)/199;h=1/199
            f=lambda s:position(d,query,s)
            now,v,a,xm,xp,times=kinematics(f,t,h)
            stress=predict_stress(d,query,times[1]).double()
            ring=torch.searchsorted(first,neighbor[:,:32])
            r,valid,parts=momentum(now,v,stress,a,g.force_sigma,local,ring)
            row={'model':directory.parent.name,'frame':frame,'coefficient':float(g.force_sigma.detach()),
                'acceleration_rms':rms(a[local][valid]),'predicted_divergence_rms':rms(parts['predicted'][valid]),
                'rate_divergence_rms':rms(parts['rate'][valid]),'residual_rms':rms(r[valid]),
                'velocity_h_sweep':[],'neighbor_sweep':[]}
            for factor in [.5,2]:
                xx,vv,aa,_,_,ts=kinematics(f,t,h*factor)
                if abs(ts[1]-t)>1e-10:
                    row['velocity_h_sweep'].append({'factor':factor,'skipped':'boundary would move center; different times are not a derivative-sensitivity comparison'})
                    continue
                rr,good,_=momentum(xx,vv,stress,aa,g.force_sigma,local,ring);both=valid&good
                row['velocity_h_sweep'].append({'factor':factor,
                    'velocity_relative_change':rms(vv-v)/max(rms(v),1e-30),
                    'acceleration_relative_change':rms(aa-a)/max(rms(a),1e-30),
                    'momentum_relative_change':rms((rr-r)[both])/max(rms(r[both]),1e-30),
                    'paired_valid_fraction':float(both.float().mean())})
            for k in [16,32,48]:
                rr,good,pp=momentum(now,v,stress,a,g.force_sigma,local,torch.searchsorted(first,neighbor[:,:k]));both=valid&good
                row['neighbor_sweep'].append({'k':k,'valid_fraction':float(good.float().mean()),
                    'paired_valid_fraction':float(both.float().mean()),
                    'residual_rms':rms(rr[good]),'relative_change_to_k32':rms((rr-r)[both])/max(rms(r[both]),1e-30)})
            records.append(row)
    del g,d;torch.cuda.empty_cache()
assert_frozen(ROOT/'variants/coherent_momentum/code')
(REPORTS/'field_sensitivity.json').write_text(json.dumps({'status':'complete','rows':records,
    'purpose':'Post-selection diagnostics only; same prefix canonical anchors, no optimization or selection changes.',
    'limitations':'FP32 field values with FP64 differencing; no finest-step ground truth. Changing neighborhood changes the estimator. Common points may have been pruned.'},indent=2)+'\n',encoding='utf8')
print('Read-only field decomposition and sensitivity saved')
