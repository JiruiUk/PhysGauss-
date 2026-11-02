"""Same explicit option-branch checks, with independently receipted new state."""
from variants.flow_comparison.checkpoint import save_training_state, restore_training_state
import copy
from pathlib import Path
import shutil
import torch
from variants.metric_tuning.run import read_json, write_json, sha256
from .model import stress_head

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT/'runs/pidg_color_repair/shared_prefix/rubber_duck'
PREFIX = ROOT/'runs/pidg_unified_flow/shared_prefix/rubber_duck'


def extend_state(state):
    assert not any(k.startswith('shared_stress.') for k in state['deform'])
    width = state['deform']['gaussian_warp.weight'].shape[1]
    additions = {'shared_stress.'+k: v for k, v in stress_head(width).state_dict().items()}
    state['deform'].update(additions)
    groups = state['deform_optimizer']['param_groups']
    indices = [p for g in groups for p in g['params']]
    assert len(indices) == len(set(indices))
    group = copy.deepcopy(groups[0])
    group.update(name='shared_stress', lr=state['opt']['velocity_net_lr'],
        params=list(range(max(indices)+1, max(indices)+1+len(additions))))
    groups.append(group)
    state['dataset']['network_args'] = dict(state['dataset']['network_args'], unified_stress_head=True)
    return additions


def convert_prefix():
    receipt = PREFIX/'conversion.json'
    original = ORIGINAL/'checkpoints/resume.pt'
    digest = sha256(original)
    if receipt.exists():
        r = read_json(receipt)
        assert r['original_sha256'] == digest and sha256(PREFIX/'checkpoints/resume.pt') == r['converted_sha256']
        return PREFIX
    assert read_json(original.with_suffix('.json'))['iteration'] == 3000
    PREFIX.mkdir(parents=True, exist_ok=True)
    for folder in ['point_cloud', 'deform', 'checkpoints']:
        shutil.copytree(ORIGINAL/folder, PREFIX/folder, dirs_exist_ok=True)
    for name in ['metric_best.json', 'metric_history.jsonl', 'model_config.py']:
        shutil.copyfile(ORIGINAL/name, PREFIX/name)
    state = torch.load(original, map_location='cpu', weights_only=False)
    additions = extend_state(state)
    target = PREFIX/'checkpoints/resume.pt'
    temporary = target.with_suffix('.tmp'); torch.save(state, temporary); temporary.replace(target)
    converted = sha256(target)
    metadata = read_json(original.with_suffix('.json'))
    metadata.update(path=str(target), sha256=converted, size_bytes=target.stat().st_size)
    write_json(target.with_suffix('.json'), metadata)
    sidecar = read_json(ORIGINAL/'checkpoints/resume.metrics.json')
    sidecar['checkpoint_sha256'] = converted
    write_json(target.with_suffix('.metrics.json'), sidecar)
    for path in (PREFIX/'deform').glob('*/deform.pth'):
        grid, net = torch.load(path, map_location='cpu', weights_only=True)
        assert not any(k.startswith('shared_stress.') for k in net)
        net.update(additions); torch.save((grid, net), path)
    write_json(receipt, {'original': str(original), 'original_sha256': digest, 'converted_sha256': converted,
        'new_parameters': {k: list(v.shape) for k, v in additions.items()}, 'head_init_seed': 20261102,
        'head_init_std': .001, 'head_bias': 0., 'extra_optimization_steps': 0,
        'changes': ['add shared_stress Linear(256,6) state', 'append empty-state optimizer group at original velocity LR',
                    'declare optional head architecture in saved dataset', 'extend inference snapshots'],
        'original_tensors_optimizer_state_and_RNG_preserved': True,
        'both_arms_copy_same_converted_checkpoint_bytes': True})
    return PREFIX
