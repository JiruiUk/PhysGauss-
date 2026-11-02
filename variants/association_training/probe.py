"""Real C1 complete-render gradients and an isolated optimizer update."""
import json
from pathlib import Path
import torch
import numpy as np

from variants.association_training.run import ROOT, REPORT, RUNS, SOURCE, configuration
from variants.association_training.prepare_data import prepare
from variants.association_training.runtime import Tuning
from variants.metric_tuning.runtime import read_json, write_json


def main():
    torch.set_num_threads(4)
    binding = prepare()
    config = configuration(True, binding)
    folder = RUNS / '_probe'
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'metric_config.json', config)
    from variants.c1_flow_ablation.results import motion_module
    m = motion_module()
    g, d = m.load(SOURCE, '50000')
    t = Tuning(folder / 'metric_config.json', folder)
    t.association_deform = d
    with np.load(binding['path']) as z:
        t.association_labels = torch.tensor(z['labels'], device='cuda', dtype=torch.long)
        t.anchor_frames = z['anchor_frames'].tolist()
        t.anchor_masks = z['anchor_masks'].copy()
        t.anchor_reference = z['anchor_reference'].copy()
    t.metadata = read_json(ROOT / 'data/derived/kinematic_resplit/training/transforms_train.json')
    t.frames = {int(Path(f['file_path']).stem): f for f in t.metadata['frames']}
    with np.load(ROOT / 'output/surface_support/image_tracks.npz') as z:
        t.components = {int(n): c.copy() for n, c in zip(z['frames'], z['components'])}
    t.flow_root = ROOT / 'data/derived/flow_comparison_train_only'
    t.flow_rows = read_json(t.flow_root / 'manifest.json')['pairs']
    loss = t.association_loss(50020, g)
    groups = {'grid': list(d.grid4d.parameters()), 'deformation': list(d.deform.parameters()),
              'canonical_centers': [g._xyz], 'opacity': [g._opacity], 'velocity_stress': list(g.velocity_net.parameters())}
    gradients = {}
    for key, params in groups.items():
        grad = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        assert all(torch.isfinite(v).all() for v in grad if v is not None)
        gradients[key] = sum(float(v.abs().sum()) for v in grad if v is not None)
    assert all(gradients[k] > 0 for k in ['grid', 'deformation', 'canonical_centers', 'opacity'])
    assert gradients['velocity_stress'] == 0
    params = [p for key, group in groups.items() if key != 'velocity_stress' for p in group]
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.Adam(params, lr=1e-6)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    changes = sum(float((p.detach() - b).abs().sum()) for p, b in zip(params, before))
    assert changes > 0
    result = {'status': 'passed', 'complete_native_render_width': 1600, 'loss': float(loss.detach()),
              'finite_gradient_L1': gradients, 'actual_parameter_L1_change': changes,
              'historical_checkpoint_written': False, 'new_identity_parameters': False}
    write_json(REPORT / 'optimizer_probe.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
