"""Post-selection independent image motion and common-anchor dual operators."""
import importlib.util
import os
import sys
import torch
from .run import ROOT,REPORTS,RUNS,NAME
from .operator import kinematics,momentum
from variants.metric_tuning.run import read_json as read,write_json as write,sha256
from variants.flow_comparison import temporal
from variants.flow_comparison.local_momentum import Graph,residual
from variants.unified_flow.model import predict_stress
CONTROL=ROOT/'runs/pidg_momentum_guard/M1_guarded_shared/rubber_duck'


def motion_module():
    sys.path.insert(0,str(ROOT/'variants/coherent_momentum/code'))
    from scene_PIDG.deform_model import DeformModel
    import inspect
    assert 'coherent_momentum' in inspect.getfile(DeformModel)
    spec=importlib.util.spec_from_file_location('coherent_motion',ROOT/'output/flow_comparison/evaluate_motion_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def common_probes(loader):
    prefix=ROOT/'runs/pidg_unified_flow/shared_prefix/rubber_duck/checkpoints/resume.pt'
    state=torch.load(prefix,map_location='cpu',weights_only=False)
    canonical=state['gaussian_parameters']['_xyz'].cuda();del state
    graphs={k:Graph().sample(canonical,50000,count=64,k=k) for k in [12,32]}
    rows=[]; tensors={}
    for directory in [CONTROL,RUNS/NAME/'rubber_duck']:
        os.environ['PIDG_METRIC_CONFIG']=str(directory/'metric_config.json')
        g,d=loader(directory,'best')
        with torch.no_grad():
            for frame in [2,83,155]:
                t=(frame-1)/199
                for operator in ['legacy','coherent']:
                    all_ids,first,anchors,nf,na=graphs[12 if operator=='legacy' else 32]
                    if operator=='legacy':
                        x=canonical[all_ids];f=lambda s:temporal.position(d,x,s)
                        now=f(t);v=temporal.derivative(f,t,.001)
                        a=temporal.derivative(f,t,1/199,2);stress=predict_stress(d,x,t)
                        r,valid=residual(now,v,stress,a,g.force_sigma,all_ids,first,anchors,nf,na)
                        h=1/199
                    else:
                        x=canonical[first];f=lambda s:temporal.position(d,x,s)
                        now,v,a,xm,xp,times=kinematics(f,t,1/199)
                        stress=predict_stress(d,x,times[1]).double()
                        loc=torch.searchsorted(first,anchors);ring=torch.searchsorted(first,na)
                        r,valid,parts=momentum(now,v,stress,a,g.force_sigma,loc,ring)
                        h=1/199
                    key=(directory.parent.name,frame,operator)
                    tensors[key]=(r.detach().double().cpu(),valid.detach().cpu())
                    rows.append({'model':key[0],'frame':frame,'operator':operator,
                        'valid_fraction':float(valid.float().mean()),'valid_anchors':int(valid.sum()),
                        'raw_huber':float(torch.nn.functional.huber_loss(r[valid],torch.zeros_like(r[valid]))),
                        'residual_rms':float(r[valid].double().square().mean().sqrt()),
                        'impulse_rms':float(h*r[valid].double().square().mean().sqrt()),
                        'rollout_position_rms':float(h*h*r[valid].double().square().mean().sqrt())})
        del g,d;torch.cuda.empty_cache()
    comparisons=[]
    for operator in ['legacy','coherent']:
        hs={m:[] for m in ['M1_guarded_shared',NAME]};coverage=[]
        for frame in [2,83,155]:
            old,vo=tensors[('M1_guarded_shared',frame,operator)]
            new,vn=tensors[(NAME,frame,operator)];valid=vo&vn
            coverage.append(float(valid.float().mean()))
            if not valid.any(): raise ValueError('No paired valid anchors')
            for model,r in [('M1_guarded_shared',old),(NAME,new)]:
                hs[model].append(float(torch.nn.functional.huber_loss(r[valid],torch.zeros_like(r[valid]))))
        import statistics
        a,b=[statistics.median(hs[m]) for m in ['M1_guarded_shared',NAME]]
        comparisons.append({'operator':operator,'matched_huber_by_frame':hs,'M1_median':a,'N1_median':b,
            'relative_change':b/max(a,1e-30)-1,'min_paired_valid_fraction':min(coverage)})
    result={'status':'complete','rows':rows,'paired_comparisons':comparisons,
        'prefix_sha256':sha256(prefix),'frames':[2,83,155],
        'meaning':'same common canonical anchors; each operator applied identically to both models; residuals from different operators are not interchangeable',
        'limitations':'No physical ground truth. Prefix points may have been pruned; current geometry/conditioning differs. Rollout error is scaled residual, not independent validation.'}
    write(REPORTS/'common_momentum.json',result)
    return result


def evaluate():
    module=motion_module();manifest=read(module.CACHE/'manifest.json')
    assert all(sha256(module.CACHE/k)==v for k,v in manifest['output_sha256'].items())
    assert (RUNS/'selection_lock.json').exists()
    run=RUNS/NAME/'rubber_duck'
    rows=[module.evaluate(run,tag,manifest) for tag in ['best','50000']]
    write(REPORTS/'motion_validation.json',{'status':'complete','results':rows,
        'manifest_sha256':sha256(module.CACHE/'manifest.json'),
        'comparison_velocity_h':.001,'note':'Frozen V2 h=.001 in both old/new models; actual N1 training h=1/199. Finite-difference sensitivity is reported, no physical GT.'})
    return rows,common_probes(module.load)
