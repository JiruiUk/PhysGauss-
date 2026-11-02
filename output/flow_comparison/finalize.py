"""Verify frozen experiment receipts and build comparison figures after training."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from variants.flow_comparison.run import assert_frozen
from variants.flow_comparison.prepare_data import prepare as verify_data
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
REPORTS = ROOT/'experiments/flow_comparison'; RUNS = ROOT/'runs/pidg_flow_comparison'; OUT = Path(__file__).parent
names = ['K1_flow_independent', 'K2_flow_trajectory']
assert read(REPORTS/'progress.json')['status'] == 'completed'
assert_frozen(ROOT/'variants/flow_comparison/code'); verify_data()
prefix = ROOT/'runs/pidg_color_repair/shared_prefix/rubber_duck/checkpoints/resume.pt'
prefix_sha = read(prefix.with_suffix('.json'))['sha256']; assert sha(prefix) == prefix_sha
old = torch.load(prefix, map_location='cpu', weights_only=False)
configs = [read(RUNS/n/'rubber_duck/metric_config.json') for n in names]
assert {k for k in set(configs[0])|set(configs[1]) if configs[0].get(k) != configs[1].get(k)} == {'velocity_source'}
checks = {}; schedules = []
for name in names:
    run = RUNS/name/'rubber_duck'; branch = read(run/'branch.json')
    assert branch['identical_full_checkpoint_sha256'] == prefix_sha
    checkpoint = run/'checkpoints/resume.pt'; manifest = read(checkpoint.with_suffix('.json'))
    assert manifest['iteration'] == 50000 and sha(checkpoint) == manifest['sha256']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    net = state['velocity_net']; frozen = [k for k in net if k.startswith(('time_embed.', 'time_proj.'))]
    if name == names[1]: frozen += [k for k in net if k.startswith('velocity_head.')]
    assert all(torch.equal(net[k], old['velocity_net'][k]) for k in frozen)
    assert not torch.equal(net['stress_head.weight'], old['velocity_net']['stress_head.weight'])
    assert all(torch.isfinite(v).all() for group in ['gaussian_parameters', 'velocity_net', 'grid4d', 'deform'] for v in state[group].values())
    flow = [json.loads(s) for s in (run/'observed_flow.jsonl').read_text().splitlines()]
    momentum = [json.loads(s) for s in (run/'material_momentum.jsonl').read_text().splitlines()]
    pairs = {(p['source'], p['target']): p for p in read(REPORTS/'flow_data_manifest.json')['pairs']}
    assert all((r['source'], r['target']) in pairs and abs(r['dt']-(r['target']-r['source'])/199)<1e-14 for r in flow)
    assert len(momentum) == 420 and all(0 < r['valid_fraction'] <= 1 for r in momentum)
    schedules.append([(r['iteration'], r['source'], r['target']) for r in flow])
    export = read(REPORTS/(name+'_exports.json'))
    assert export['pairs'] == 100 and export['pixel_identity_verified'] and export['independent_PSNR_max_difference'] < 1e-4
    # The inherited trainer evaluates losses but skips optimizer.step at its
    # final iteration. Count evaluations separately from applied updates.
    checks[name] = {'final_checkpoint_sha256': manifest['sha256'], 'flow_loss_evaluations': len(flow),
        'flow_applied_update_iterations': sum(r['iteration'] < 50000 for r in flow),
        'momentum_loss_evaluations': len(momentum),
        'momentum_applied_update_iterations': sum(r['iteration'] < 50000 for r in momentum),
        'min_valid_neighborhood_fraction': min(r['valid_fraction'] for r in momentum),
        'frozen_parameter_keys_verified': frozen, 'all_final_parameters_finite': True,
        'export_pixel_and_metric_checks': True}
    del state
assert schedules[0] == schedules[1]
del old
fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
for parent, name, label in [('pidg_integrity_tuning', 'H3_combined', 'H3 historical'),
        ('pidg_monotonic_training', 'J1_monotonic_kinematic', 'J1 historical'),
        ('pidg_flow_comparison', names[0], 'K1 independent + flow'), ('pidg_flow_comparison', names[1], 'K2 trajectory + flow')]:
    run = ROOT/'runs'/parent/name/'rubber_duck'
    history = [json.loads(s) for s in (run/'metric_history.jsonl').read_text().splitlines()]
    axes[0].plot([r['iteration'] for r in history], [r['metrics']['PSNR'] for r in history], label=label)
    diagnostics = [json.loads(s) for s in (run/'kinematic_diagnostics.jsonl').read_text().splitlines()]
    axes[1].plot([r['iteration'] for r in diagnostics], [np.mean([v['relative_rmse'] for v in r['times']])*100 for r in diagnostics], label=label)
axes[0].set(xlabel='Total iteration', ylabel='Validation PSNR (dB)', title='Reconstruction (validation only)')
axes[1].set(xlabel='Total iteration', ylabel='Internal relative velocity RMSE (%)', title='K2 zero is imposed by definition')
axes[1].set_ylim(-5, min(250, axes[1].get_ylim()[1]))
for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.savefig(OUT/'training_comparison.png', dpi=180); plt.close(fig)
motion = read(REPORTS/'motion_validation.json')
fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
best = [r for r in motion['results'] if r['snapshot'] == 'best']
for offset, key, label in [(-.24, 'camera_only_static_EPE_px', 'Camera only / static object'),
        (0, 'geometric_EPE_px', 'Deformed trajectory'), (.24, 'velocity_EPE_px', 'Velocity advection')]:
    ax.bar(np.arange(2)+offset, [r['metrics'][key] for r in best], width=.23, label=label)
ax.set_xticks(np.arange(2), ['K1 independent + flow', 'K2 trajectory + flow'])
ax.set(ylabel='Mean endpoint error (pixels)', title='Post-selection validation motion: estimated raw optical flow')
ax.grid(axis='y', alpha=.2); ax.legend(fontsize=8)
fig.savefig(OUT/'motion_comparison.png', dpi=180); plt.close(fig)
receipt = {'status': 'passed', 'same_full_prefix_sha256': prefix_sha, 'only_arm_config_difference': 'velocity_source',
    'identical_flow_frame_schedule': True, 'source_and_target_hashes_verified': True, 'models': checks,
    'analysis_scripts_sha256': {p.name: sha(p) for p in OUT.glob('*.py')}}
(REPORTS/'final_verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
