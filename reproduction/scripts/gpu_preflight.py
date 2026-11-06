import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

project = Path(__file__).resolve().parents[2]
repo = project / "third_party/PIDG"
sys.path.insert(0, str(repo))
os.chdir(repo)
from hashencoder.hashgrid import HashEncoder
from motion_utils.pidg import VelocityNet, cauchy_momentum_constraint
from scene_PIDG import GaussianModel
from scene_PIDG.cameras import Camera
from utils.graphics_utils import BasicPointCloud
from gaussian_renderer import render
from argparse import Namespace

torch.manual_seed(0)
encoder = HashEncoder(input_dim=3, num_levels=2, level_dim=2, base_resolution=4,
                      desired_resolution=8, log2_hashmap_size=8).cuda()
x = torch.rand(16, 3, device="cuda", requires_grad=True)
encoded = encoder(x)
encoded.square().mean().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()

material = VelocityNet(max_particles=32).cuda()
coordinates = torch.rand(16, 4, device="cuda", requires_grad=True)
indices = torch.arange(16, device="cuda")
loss = cauchy_momentum_constraint(coordinates, lambda c: material(c, indices), torch.ones(1, device="cuda"))
loss.backward()
assert torch.isfinite(loss) and material.velocity_head.weight.grad is not None

gaussians = GaussianModel(3)
rng = np.random.default_rng(0)
points = rng.uniform(-0.4, 0.4, (32, 3)).astype(np.float32)
cloud = BasicPointCloud(points=points, colors=np.full_like(points, 0.7), normals=np.zeros_like(points))
gaussians.create_from_pcd(cloud, 1.0)
camera = Camera(0, np.eye(3), np.array([0., 0., 3.]), 0.8, 0.8,
                torch.zeros(3, 32, 32), None, "preflight", 0, fid=0.0)
deformation = torch.zeros(32, 7, device="cuda")
deformation[:, 0] = 1
result = render(camera, gaussians, Namespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False),
                torch.zeros(3, device="cuda"), deformation, torch.zeros(32, 4, device="cuda"),
                torch.zeros(32, 3, device="cuda"), True)
result["render"].square().mean().backward()
assert torch.isfinite(result["render"]).all() and gaussians._xyz.grad is not None
assert torch.isfinite(gaussians._xyz.grad).all()
torch.cuda.synchronize()
report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
          "hash_encoder_gradient": "passed", "material_second_order_gradient": "passed",
          "official_flow_rasterizer_gradient": "passed", "cmr_loss": float(loss.detach()),
          "max_cuda_memory_mb": torch.cuda.max_memory_allocated()/1024**2}
(project / "reproduction/gpu_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
