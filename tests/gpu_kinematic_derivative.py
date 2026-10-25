"""Read-only CUDA derivative audit on a frozen parent, outside the training queue."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO = ROOT/'variants/kinematic_tuning/code'
sys.path.insert(0, str(REPO))
os.environ['PIDG_HASH_SOURCE_DIR'] = str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR'] = str(ROOT/'third_party/PIDG/tmp_build')


def main():
    import numpy as np
    import torch
    from plyfile import PlyData
    from scene_PIDG.deform_model import DeformModel
    from variants.kinematic_tuning.model import center_trajectory, trajectory_velocity
    from variants.metric_tuning.run import write_json, sha256
    args = argparse.ArgumentParser()
    args.add_argument('--output', type=Path, default=ROOT/'experiments/kinematic_tuning/derivative_audit.json')
    args = args.parse_args()
    parent = ROOT/'runs/pidg_official/rubber_duck'
    config = runpy.run_path(str(REPO/'arguments/pidg/rubber_dack.py'))
    model = DeformModel(config['grid_args'], config['network_args'], reg_spatial_able=False, reg_temporal_able=False)
    model.load_weights(str(parent), 'best')
    points = PlyData.read(str(parent/'point_cloud/iteration_best/point_cloud.ply'))['vertex']
    x = np.stack([points[k] for k in ('x', 'y', 'z')], -1)
    x = torch.from_numpy(x[np.linspace(0, len(x)-1, 128).astype(int)].copy()).cuda()
    t = torch.linspace(.12, .88, len(x), device='cuda').reshape(-1, 1).requires_grad_(True)
    def trajectory(x, time):
        return center_trajectory(x, model.step(x, time)['d_xyz'])
    positions = trajectory(x, t)
    exact = torch.cat([torch.autograd.grad(positions[:, i].sum(), t, retain_graph=True)[0] for i in range(3)], -1).detach()
    assert torch.isfinite(exact).all()
    result = {'points': len(x), 'parent_deformation_sha256': sha256(parent/'deform/iteration_best/deform.pth'),
              'reference': 'first derivative from CUDA encoder input backward; no second derivative requested',
              'reference_velocity_rms': exact.square().mean().sqrt().item(), 'finite_differences': []}
    with torch.no_grad():
        for h in (.002, .001, .0005, .00025):
            numeric = trajectory_velocity(trajectory, x, t.detach(), h)
            assert torch.isfinite(numeric).all()
            rmse = (numeric-exact).square().mean().sqrt().item()
            result['finite_differences'].append({'h': h, 'rmse_against_autodiff': rmse,
                'relative_rmse': rmse/max(result['reference_velocity_rms'], 1e-8)})
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
