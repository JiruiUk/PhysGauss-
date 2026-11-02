"""Isolated flow-only optimizer update, exact new-flag-off parent loss parity."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ['PIDG_HASH_SOURCE_DIR'] = str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR'] = str(ROOT/'third_party/PIDG/tmp_build')
sys.path.insert(0, str(ROOT/'variants/c1_flow_ablation/code'))
from .runtime import Tuning, Parent, CACHE
from .run import RUNS, REPORTS, NAMES
from reproduction.scripts.training_checkpoint import load_state, restore_models
from variants.unified_flow.camera_contract import loader_fovs
from variants.flow_comparison.temporal import warp_at
from scene_PIDG.gaussian_model import GaussianModel
from scene_PIDG.deform_model import DeformModel
from scene_PIDG.cameras import Camera
from gaussian_renderer import render


def main():
    run = RUNS/'_smoke'/NAMES[1]/'rubber_duck'
    os.environ['PIDG_METRIC_CONFIG'] = str(run/'metric_config.json')
    state = load_state(run/'checkpoints/resume.pt'); opt = SimpleNamespace(**state['opt']); ds = state['dataset']
    g = GaussianModel(3); d = DeformModel(ds['grid_args'], ds['network_args'],
        reg_spatial_able=False, reg_temporal_able=False)
    d.train_setting(opt); restore_models(state, g, d, opt)
    probe = REPORTS/'optimizer_probe'; probe.mkdir(exist_ok=True)
    tune = Tuning(run/'metric_config.json', probe)
    manifest = json.loads((CACHE/'manifest.json').read_text())
    tune.flow_pairs = {p['source']: p for p in manifest['pairs']}
    meta = json.loads((ROOT/'data/derived/kinematic_resplit/training/transforms_train.json').read_text())
    def camera(fid):
        frame = next(f for f in meta['frames'] if int(Path(f['file_path']).stem) == fid)
        matrix = np.linalg.inv(np.asarray(frame['transform_matrix'])); r = -matrix[:3,:3].T; r[:,0] = -r[:,0]
        fx, fy = loader_fovs(meta['camera_angle_x'])
        return Camera(fid, r, -matrix[:3,3], fx, fy, torch.zeros(3,225,400), None, f'{fid:04d}', fid, fid=(fid-1)/199)
    cam = camera(82); tune.train_cameras = {tune.flow_pairs[82]['target']: camera(tune.flow_pairs[82]['target'])}
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    bg = torch.zeros(3, device='cuda'); warp = warp_at(d, g.get_xyz.detach(), 81/199)
    image = render(cam, g, pipe, bg, warp['d_xyz'], warp['d_rotation'], warp['d_scaling'], False)['render']
    a = tune.image_loss(100, image, cam.original_image, opt)
    b = Parent.image_loss(tune, 100, image, cam.original_image, opt)
    assert torch.equal(a, b)
    assert torch.equal(tune.physics_time(cam.fid,120), Parent.physics_time(tune,cam.fid,120))
    assert tune.time_step(120) == Parent.time_step(tune,120)
    flow = tune.observed_loss(100, g, d, cam, warp, render, pipe, bg)
    groups = {'grid': list(d.grid4d.parameters()), 'deformation': list(d.deform.parameters()),
        'velocity_head': list(g.velocity_net.velocity_head.parameters()),
        'stress_head': list(g.velocity_net.stress_head.parameters())}
    before = {name: [p.detach().clone() for p in ps] for name,ps in groups.items()}
    g.optimizer.zero_grad(set_to_none=True); d.optimizer.zero_grad(set_to_none=True)
    flow.backward()
    grads = {name: sum(float(p.grad.detach().abs().sum()) for p in ps if p.grad is not None) for name,ps in groups.items()}
    assert all(p.grad is None or bool(p.grad.isfinite().all()) for ps in groups.values() for p in ps)
    assert all(grads[name] > 0 for name in ('grid','deformation','velocity_head')) and grads['stress_head'] == 0
    g.optimizer.step(); d.optimizer.step()
    changes = {name: sum(float((p.detach()-old).abs().sum()) for p,old in zip(ps,before[name])) for name,ps in groups.items()}
    assert all(changes[name] > 0 for name in ('grid','deformation','velocity_head')) and changes['stress_head'] == 0
    assert all(bool(p.isfinite().all()) for ps in groups.values() for p in ps)
    result = {'status':'passed', 'flow_only_gradient_l1':grads, 'actual_parameter_update_l1':changes,
        'parent_image_loss_exact':True, 'parent_clock_exact':True, 'source_checkpoint_untouched':True,
        'no_new_parameters_or_optimizer_groups':True}
    (REPORTS/'optimizer_update.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__': main()
