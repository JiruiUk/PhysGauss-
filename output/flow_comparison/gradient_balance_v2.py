"""Read-only weighted gradient probes on validation-selected snapshots.

Three fixed training times are local diagnostics, not a replay of every actual
stochastic training batch. No optimizer steps or checkpoint writes occur.
"""
import json
import os
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).parent))
import evaluate_motion_v2 as motion
from variants.flow_comparison.runtime import Tuning, CACHE
from variants.flow_comparison.temporal import warp_at

ROOT = motion.ROOT
rows = []
assert (motion.RUNS/'selection_lock.json').exists()
for name in ['K1_flow_independent', 'K2_flow_trajectory']:
    run = motion.RUNS/name/'rubber_duck'
    os.environ['PIDG_METRIC_CONFIG'] = str(run/'metric_config.json')
    gaussians, deform = motion.load(run, 'best')
    tuning = Tuning(run/'metric_config.json', run)
    manifest = motion.read(CACHE/'manifest.json')
    tuning.flow_pairs = {p['source']: p for p in manifest['pairs']}; tuning.flow_tensors = {}
    # Loss methods log diagnostics to a dedicated directory, not the original
    # training log. This preserves the distinction between probes and updates.
    tuning.run = ROOT/'output/flow_comparison/gradient_probes_v2'/name
    tuning.run.mkdir(parents=True, exist_ok=True)
    groups = {'deformation': [p for p in deform.deform.parameters() if p.requires_grad],
        'velocity_head': [p for p in gaussians.velocity_net.velocity_head.parameters() if p.requires_grad],
        'stress_head': [p for p in gaussians.velocity_net.stress_head.parameters() if p.requires_grad],
        'physics_shared': [p for p in gaussians.velocity_net.shared_net.parameters() if p.requires_grad],
        'rate_coefficient': [gaussians.force_sigma]}
    params = [p for values in groups.values() for p in values]
    for frame in [2, 83, 155]:
        cam = motion.camera(motion.FRAMES[frame]); t = (frame-1)/199
        raw_momentum = tuning.momentum_loss(50000, gaussians, deform, cam.fid)
        raw_flow = tuning.motion_loss(50000, gaussians, deform, cam, warp_at(deform, gaussians.get_xyz.detach(), t),
            motion.render, motion.PIPE, motion.BACKGROUND)
        gm = torch.autograd.grad(.01*raw_momentum, params, allow_unused=True, retain_graph=True)
        gf = torch.autograd.grad(.01*raw_flow, params, allow_unused=True)
        offset = 0; record = {'model': name, 'snapshot': 'best', 'training_frame': frame,
            'weighted_momentum_loss': float(.01*raw_momentum.detach()), 'weighted_flow_loss': float(.01*raw_flow.detach()),
            'groups': {}}
        for key, values in groups.items():
            pairs = list(zip(values, gm[offset:offset+len(values)], gf[offset:offset+len(values)])); offset += len(values)
            nm = nf = dot = 0.
            for parameter, a, b in pairs:
                if a is not None:
                    assert a.isfinite().all(); nm += float(a.detach().double().square().sum())
                if b is not None:
                    assert b.isfinite().all(); nf += float(b.detach().double().square().sum())
                if a is not None and b is not None:
                    dot += float((a.detach().double()*b.detach().double()).sum())
            nm, nf = nm**.5, nf**.5
            record['groups'][key] = {'momentum_norm': nm, 'flow_norm': nf,
                'momentum_to_flow_norm_ratio': nm/nf if nf>1e-20 else None,
                'cosine': dot/(nm*nf) if min(nm, nf)>1e-20 else None}
        rows.append(record); print(json.dumps(record), flush=True)
        del raw_flow, raw_momentum, gm, gf
    del gaussians, deform, tuning, params, groups
    torch.cuda.empty_cache()
(motion.REPORTS/'gradient_balance_v2.json').write_text(json.dumps({'status': 'complete', 'probes': rows,
    'protocol': 'Actual loader camera convention. Probe the original frozen V1 training objectives, including their known flow errors; not the corrected V2 training objective.',
    'limitations': 'local fixed-time probes; not a complete training-gradient audit; norm imbalance alone does not prove causation',
    'script_sha256': motion.sha(Path(__file__))}, indent=2))
