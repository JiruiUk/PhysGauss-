"""Read-only source/config/RNG/camera contracts and pre-intervention drift."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.momentum_guard.runtime import Tuning,Parent
from variants.metric_tuning.run import runtime_hashes,sha256,read_json as read,write_json as write
from variants.momentum_guard.run import assert_frozen
REPORTS=ROOT/'experiments/momentum_repair'
assert_frozen(ROOT/'variants/momentum_guard/code')
old=ROOT/'variants/unified_flow/code';new=ROOT/'variants/momentum_guard/code'
a,b=runtime_hashes(old),runtime_hashes(new)
assert a.keys()==b.keys();changed=[k for k in a if a[k]!=b[k]];assert changed==['train_pidg.py']
assert (new/'train_pidg.py').read_text(encoding='utf8')==(old/'train_pidg.py').read_text(encoding='utf8').replace('from variants.unified_flow.runtime import Tuning','from variants.momentum_guard.runtime import Tuning')
r=ROOT/'runs/pidg_momentum_guard/M1_guarded_shared/rubber_duck'
c=ROOT/'runs/pidg_unified_flow/L1_shared_stress/rubber_duck'
tune=Tuning(r/'metric_config.json',r)
cpu=torch.random.get_rng_state().clone();value=torch.tensor(2.,requires_grad=True)
with patch.object(Parent,'image_loss',return_value=value),patch.object(Parent,'momentum_loss',return_value=value):
    for step in [3001,4000,7999,8000]:
        assert tune.image_loss(step,None,None,None) is value
        assert tune.momentum_loss(step,None,None,None) is value
        assert tune._reference is None
assert torch.equal(cpu,torch.random.get_rng_state())
assert read(r/'branch.json')['identical_full_checkpoint_sha256']==read(c/'branch.json')['identical_full_checkpoint_sha256']
states=[torch.load(d/'checkpoints/resume.pt',map_location='cpu',weights_only=False) for d in [c,r]]
def finite_tree(value):
    if torch.is_tensor(value):return bool(value.isfinite().all())
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(finite_tree(v) for v in value)
    return True
assert all(finite_tree(s[k]) for s in states for k in ['gaussian_optimizer','deform_optimizer'])
assert states[0]['train_camera_names']==states[1]['train_camera_names']
assert states[0]['test_camera_names']==states[1]['test_camera_names']
assert states[0]['rng_python']==states[1]['rng_python']
options={k:[states[0]['opt'].get(k),states[1]['opt'].get(k)] for k in set(states[0]['opt'])|set(states[1]['opt']) if states[0]['opt'].get(k)!=states[1]['opt'].get(k)}
assert set(options)=={'save_image_dir'}
configs=[read(d/'metric_config.json') for d in [c,r]]
differences={k:[configs[0].get(k),configs[1].get(k)] for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k)}
assert set(differences)=={'momentum_gradient_guard','momentum_shared_ratio'}
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
hs=[rows(d/'metric_history.jsonl') for d in [c,r]];prior={v['iteration']:v for v in hs[0]}
drift=[{'iteration':v['iteration'],'PSNR_delta':v['metrics']['PSNR']-prior[v['iteration']]['metrics']['PSNR']} for v in hs[1] if 3000<=v['iteration']<=8000]
flow=[[(v['iteration'],v['source'],v['target']) for v in rows(d/'observed_flow.jsonl')] for d in [c,r]]
assert flow[0]==flow[1]
result={'status':'contracts_passed_with_unresolved_pre_guard_metric_drift','source_difference':changed,
    'sole_source_edit':'trainer imports optional runtime wrapper','model_config_bytes_equal':(c/'model_config.py').read_bytes()==(r/'model_config.py').read_bytes(),
    'config_difference':differences,'optimizer_option_difference':options,'same_full_start_sha256':read(r/'branch.json')['identical_full_checkpoint_sha256'],
    'same_camera_names_order':True,'same_final_python_RNG':True,'same_observed_flow_schedule':True,
    'all_final_optimizer_tensors_finite':True,
    'early_wrapper_exact_parent_tensor_and_no_torch_CPU_RNG_consumption':True,'pre_guard_validation_PSNR_drift':drift,
    'same_final_torch_CPU_RNG':torch.equal(states[0]['rng_torch'],states[1]['rng_torch']),
    'same_final_cuda_RNG':[torch.equal(x,y) for x,y in zip(states[0]['rng_cuda'],states[1]['rng_cuda'])],
    'interpretation':'Source/config/start/order contracts support a controlled policy comparison, but pre-8100 drift prevents claiming the full gain is caused solely by the guard. CUDA atomic nondeterminism is a possible explanation, not demonstrated as the unique cause. No extra fitting or retraining performed.'}
write(REPORTS/'reproducibility_audit.json',result)
print(json.dumps(result,ensure_ascii=False),flush=True)
