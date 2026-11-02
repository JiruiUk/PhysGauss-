"""An actual isolated optimizer update, beyond the trainer's final-step backward.

Reads immutable smoke checkpoints and never saves over them. The inherited
trainer intentionally skips optimizer.step at its final iteration, so its
100-step smoke's first flow/physics event needs this additional update check.
"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'variants/flow_comparison/code'))
os.environ['PIDG_HASH_SOURCE_DIR'] = str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR'] = str(ROOT/'third_party/PIDG/tmp_build')
from variants.metric_tuning.checkpoint import base
from variants.flow_comparison.runtime import Tuning, CACHE
from variants.flow_comparison.temporal import warp_at
from scene_PIDG.gaussian_model import GaussianModel
from scene_PIDG.deform_model import DeformModel
from scene_PIDG.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import focal2fov, fov2focal

reports = {}
for name in ['K1_flow_independent', 'K2_flow_trajectory']:
    run = ROOT/'runs/pidg_flow_comparison/_smoke'/name/'rubber_duck'
    os.environ['PIDG_METRIC_CONFIG'] = str(run/'metric_config.json')
    state = base.load_state(run/'checkpoints/resume.pt')
    opts = SimpleNamespace(**state['opt']); dataset = state['dataset']
    g = GaussianModel(3)
    d = DeformModel(dataset['grid_args'], dataset['network_args'], reg_spatial_able=False, reg_temporal_able=False)
    d.train_setting(opts); base.restore_models(state, g, d, opts)
    tuning = Tuning(run/'metric_config.json', run)
    manifest = json.loads((CACHE/'manifest.json').read_text()); tuning.flow_pairs = {p['source']: p for p in manifest['pairs']}
    tuning.flow_tensors = {}
    meta = json.loads((ROOT/'data/derived/kinematic_resplit/training/transforms_train.json').read_text())
    frame = next(f for f in meta['frames'] if int(Path(f['file_path']).stem) == 82)
    matrix = np.linalg.inv(np.array(frame['transform_matrix'])); rotation = -matrix[:3, :3].T; rotation[:, 0] = -rotation[:, 0]
    fovx = meta['camera_angle_x']; fovy = focal2fov(fov2focal(fovx, 1600), 900)
    camera = Camera(82, rotation, -matrix[:3, 3], fovx, fovy, torch.zeros(3, 225, 400), None, '0082', 82, fid=81/199)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    groups = {'grid': d.grid4d, 'deformation': d.deform, 'stress_head': g.velocity_net.stress_head,
              'velocity_head': g.velocity_net.velocity_head}
    selected = {key: next(module.parameters()) for key, module in groups.items()}
    before = {key: value.detach().clone() for key, value in selected.items()}
    momentum = tuning.momentum_loss(100, g, d, camera.fid)
    flow = tuning.motion_loss(100, g, d, camera, warp_at(d, g.get_xyz.detach(), 81/199), render, pipe, torch.zeros(3, device='cuda'))
    g.optimizer.zero_grad(set_to_none=True); d.optimizer.zero_grad(set_to_none=True)
    loss = .01*(momentum+flow); loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for module in groups.values() for p in module.parameters())
    g.optimizer.step(); d.optimizer.step()
    changed = {key: float((value.detach()-before[key]).abs().sum()) for key, value in selected.items()}
    assert all(changed[key] > 0 for key in ['grid', 'deformation', 'stress_head'])
    assert changed['velocity_head'] > 0 if name.startswith('K1') else changed['velocity_head'] == 0
    assert all(torch.isfinite(p).all() for module in groups.values() for p in module.parameters())
    reports[name] = {'status': 'passed', 'loss': float(loss.detach()), 'parameter_update_l1': changed,
                     'source_checkpoint_untouched': True}
    del state, g, d, tuning, momentum, flow, loss, selected, before, groups
    torch.cuda.empty_cache()
target = ROOT/'experiments/flow_comparison/optimizer_update_checks.json'
target.write_text(json.dumps(reports, indent=2)+'\n')
preflight = target.with_name('preflight.json')
receipt = json.loads(preflight.read_text()); receipt['actual_optimizer_update_checks'] = reports
preflight.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(reports, indent=2), flush=True)
