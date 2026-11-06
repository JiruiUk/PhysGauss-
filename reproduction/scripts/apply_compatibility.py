"""Minimal, idempotent repairs for official PIDG on Windows/CUDA 12.8."""
import argparse
import json
from pathlib import Path


def replace(path, old, new):
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise ValueError(f"Unexpected upstream source in {path}; refusing an ambiguous patch")
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    changes = []
    # The implementation is already present in motion_utils/pidg.py in this commit.
    path = root / "scene_PIDG/gaussian_model.py"
    if replace(path, "from motion_utils.time_evolving_material_field import VelocityNet",
               "from motion_utils.pidg import VelocityNet"):
        changes.append("Fix unresolved VelocityNet import; use the released VelocityNet in pidg.py")
    path = root / "hashencoder/backend.py"
    if replace(path, "extra_cflags=['-O3', '-std=c++14', '-std=c++17'],",
               "extra_cflags=['/O2', '/std:c++17'] if os.name == 'nt' else ['-O3', '-std=c++17'],"):
        changes.append("Use native MSVC optimization/C++17 flags for hash encoder")
    if replace(path, "'-O3', '-std=c++14', '-std=c++17', '-allow-unsupported-compiler',",
               "'-O3', '-std=c++17', '-allow-unsupported-compiler',"):
        changes.append("Remove conflicting C++14 CUDA flag")
    for relative in ("submodules/simple-knn/setup.py", "submodules/flow-based-diff-gaussian-rasterization/setup.py"):
        path = root / relative
        if relative.endswith("simple-knn/setup.py"):
            changed = replace(path, '"nvcc": []', '"nvcc": ["-allow-unsupported-compiler"]')
        else:
            changed = replace(path, '"nvcc": ["-I"', '"nvcc": ["-allow-unsupported-compiler", "-I"')
        if changed:
            changes.append(f"Allow installed MSVC 14.44 host compiler in {relative}")
    # README/release says motion_mask; this source commit hard-codes resized_mask.
    path = root / "scene_PIDG/dataset_readers.py"
    if replace(path, 'mask_path  = os.path.join(path, "resized_mask", image_name + extension)',
               'mask_path  = os.path.join(path, "resized_mask", image_name + extension)\n'
               '            if not os.path.isfile(mask_path):\n'
               '                mask_path = os.path.join(path, "motion_mask", image_name + extension)\n'
               '            if not os.path.isfile(mask_path):\n'
               '                mask_path = os.path.join(path, "mask", image_name + extension)'):
        changes.append("Resolve published motion_mask/mask directories when resized_mask is absent")
    path = root / "gaussian_renderer/__init__.py"
    if replace(path, "d_scaling, enable_flow_grad,scaling_modifier=1.0",
               "d_scaling, enable_flow_grad=False,scaling_modifier=1.0"):
        changes.append("Restore default no-flow rendering for upstream evaluation call sites")
    path = root / "scene_PIDG/cameras.py"
    if replace(path, "        self.motion_mask_image = self.motion_mask_image.to(data_device)",
               "        if self.depth is not None:\n"
               "            self.depth = self.depth.to(data_device)\n"
               "        self.w2c = self.world_view_transform.T\n"
               "        # Adjacent views provide projection/time; avoid copying their full images.\n"
               "        for adjacent in (self.next_cam, self.prev_cam):\n"
               "            if adjacent is not None:\n"
               "                for field in ('world_view_transform', 'projection_matrix',\n"
               "                              'full_proj_transform', 'camera_center', 'fid'):\n"
               "                    setattr(adjacent, field, getattr(adjacent, field).to(data_device))\n"
               "                adjacent.w2c = adjacent.world_view_transform.T"):
        changes.append("Repair on-demand GPU camera transfer: missing mask attribute and adjacent-view device mismatch")
    path = root / "scene_PIDG/__init__.py"
    if replace(path, '        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))',
               '        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))\n'
               '        import torch\n'
               '        torch.save({"velocity_net": self.gaussians.velocity_net.state_dict(),\n'
               '                    "force_sigma": self.gaussians.force_sigma.detach(),\n'
               '                    "particle_indices": self.gaussians.particle_indices,\n'
               '                    "iteration": iteration}, os.path.join(point_cloud_path, "material.pt"))'):
        changes.append("Persist the trained material field and particle IDs beside Gaussian snapshots")
    path = root / "render.py"
    if replace(path, '        gaussians.velocity_net.load_weights(dataset.model_path, iteration=iteration)',
               '        material_path = os.path.join(dataset.model_path, "point_cloud",\n'
               '                                     "iteration_" + str(scene.loaded_iter), "material.pt")\n'
               '        material = torch.load(material_path, map_location="cuda", weights_only=True)\n'
               '        gaussians.velocity_net.load_state_dict(material["velocity_net"])\n'
               '        gaussians.particle_indices = material["particle_indices"]\n'
               '        gaussians.force_sigma.data.copy_(material["force_sigma"])'):
        changes.append("Replace the missing VelocityNet.load_weights method with actual saved material state")
    print(json.dumps(changes, indent=2), flush=True)


if __name__ == "__main__":
    main()
