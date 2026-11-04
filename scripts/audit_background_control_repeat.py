"""Check a repeated control instead of hiding divergent historical results."""
import json
import torch
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.background_residual.prepare import EXP, WORK, verify
from scripts.finalize_background_residual import equal_tree


def main():
    verify()
    old_exp = ROOT/'experiments/staged_reconstruction/motion_integrity'
    old_dir = ROOT/'runs/pidg_staged_reconstruction/motion_integrity/formal/I0_control'
    new_dir = WORK/'formal/I10_control'
    old_report = read(old_exp/'screen_evaluation.json')
    assert sha(old_dir/'checkpoint.pt') == old_report['models']['I0_control']['sha256']
    old_freeze = read(old_exp/'preparation_freeze.json')
    assert all(sha(ROOT/name) == expected for name, expected in old_freeze.items())
    initial_a = torch.load(old_dir/'initial.pt', map_location='cpu', weights_only=False)
    initial_b = torch.load(new_dir/'initial.pt', map_location='cpu', weights_only=False)
    fields = ['model', 'optimizer', 'rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda', 'camera_stack', 'physics_graphs']
    initial_checks = {field: equal_tree(initial_a[field], initial_b[field]) for field in fields}
    assert all(initial_checks.values())
    old = [json.loads(v) for v in (old_dir/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
    new = [json.loads(v) for v in (new_dir/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(old) == len(new) == 2000
    assert [v['frame_index'] for v in old] == [v['frame_index'] for v in new]
    first_difference = next((a['local_step'] for a, b in zip(old, new) if a['rgb'] != b['rgb']), None)
    final_a = torch.load(old_dir/'checkpoint.pt', map_location='cpu', weights_only=False)
    final_b = torch.load(new_dir/'checkpoint.pt', map_location='cpu', weights_only=False)
    final_rng_checks = {field: equal_tree(final_a[field], final_b[field]) for field in fields[2:7]}
    assert all(final_rng_checks.values())
    old_metric = old_report['validation']['I0_control']['metrics']['dynamic_PSNR']
    new_metric = read(EXP/'screen_evaluation.json')['validation']['I10_control']['metrics']['dynamic_PSNR']
    old_first_physics = next(v for v in old if 'physics' in v)
    new_first_physics = next(v for v in new if 'physics' in v)
    write(EXP/'control_repeat_audit.json', {
        'historical_control_sha256': sha(old_dir/'checkpoint.pt'), 'new_control_sha256': sha(new_dir/'checkpoint.pt'),
        'historical_initial_sha256': sha(old_dir/'initial.pt'), 'new_initial_sha256': sha(new_dir/'initial.pt'),
        'initial_state_checks': initial_checks, 'historical_frozen_source_hashes_match': True,
        'all2000_camera_samples_match': True, 'final_RNG_checks': final_rng_checks,
        'first_nonidentical_logged_RGB_step': first_difference,
        'early_RGB': [{'step': a['local_step'], 'historical': a['rgb'], 'new': b['rgb'], 'delta': b['rgb']-a['rgb']} for a, b in zip(old[:10], new[:10])],
        'first_physics_local_step': old_first_physics['local_step'],
        'first_physics_unique_node_counts': [old_first_physics['physics']['diagnostics'][0]['nodes'], new_first_physics['physics']['diagnostics'][0]['nodes']],
        'historical_dynamic_PSNR': old_metric, 'new_dynamic_PSNR': new_metric, 'difference_dB': new_metric-old_metric,
        'limit': 'Two controls demonstrate numerical trajectory sensitivity, not a distribution or confidence interval. CUDA atomic/numerical variation is plausible, not a uniquely established root cause.',
        'implication': 'A single candidate-control PSNR difference cannot establish a stable causal effect. Actual visible fragmentation independently fails completeness.',
        'script_sha256': sha(__file__), 'PIDG_optimizer_updates': 0})
    print('BACKGROUND_CONTROL_REPEAT_AUDITED', first_difference, new_metric-old_metric)


if __name__ == '__main__':
    main()
