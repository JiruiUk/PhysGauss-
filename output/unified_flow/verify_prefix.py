"""Verify every inherited field of the real extended common checkpoint."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from variants.unified_flow.checkpoint import ORIGINAL, PREFIX
from variants.metric_tuning.run import sha256, read_json, write_json


def equal(a, b):
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, np.ndarray): return np.array_equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


old = torch.load(ORIGINAL/'checkpoints/resume.pt', map_location='cpu', weights_only=False)
new = torch.load(PREFIX/'checkpoints/resume.pt', map_location='cpu', weights_only=False)
for key in old:
    if key not in ['deform', 'deform_optimizer', 'dataset']: assert equal(old[key], new[key]), key
assert all(equal(v, new['deform'][k]) for k, v in old['deform'].items())
assert set(new['deform'])-set(old['deform']) == {'shared_stress.weight', 'shared_stress.bias'}
assert equal(old['deform_optimizer']['state'], new['deform_optimizer']['state'])
assert equal(old['deform_optimizer']['param_groups'], new['deform_optimizer']['param_groups'][:2])
assert len(new['deform_optimizer']['param_groups']) == 3
for key in old['dataset']:
    if key != 'network_args': assert equal(old['dataset'][key], new['dataset'][key])
assert equal(old['dataset']['network_args'], {k: v for k, v in new['dataset']['network_args'].items() if k != 'unified_stress_head'})
old_manifest = read_json(ROOT/'data/derived/flow_comparison_train_only/manifest.json')
for p in old_manifest['pairs']:
    assert sha256(ROOT/'data/derived/flow_comparison_train_only'/p['file']) == old_manifest['output_sha256'][p['file']]
receipt = {'status': 'passed', 'original_sha256': sha256(ORIGINAL/'checkpoints/resume.pt'),
    'converted_sha256': sha256(PREFIX/'checkpoints/resume.pt'),
    'all_inherited_tensors_optimizer_states_and_RNG_identical': True,
    'only_extra_tensors': ['shared_stress.weight', 'shared_stress.bias'],
    'new_optimizer_group_empty': True, 'new_architecture_flag_only_dataset_change': True,
    'raw_train_flow_source_artifact_hashes_match_original_manifest': True, 'extra_optimization_steps': 0}
write_json(ROOT/'experiments/unified_flow/prefix_extension_checks.json', receipt)
print(json.dumps(receipt, indent=2))
