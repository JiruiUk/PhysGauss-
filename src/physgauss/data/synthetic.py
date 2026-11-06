from pathlib import Path
import json
import math
import numpy as np
import torch
from ..models.gaussians import GaussianState
from ..rendering import Camera, ReferenceRenderer


def make_synthetic(output, count=9, size=24, particles=12):
    """Small moving ellipsoids with known cameras, depth and projected particle flow."""
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    if count < 3 or size < 8 or particles < 2:
        raise ValueError("Synthetic scene needs frames>=3, size>=8, particles>=2")
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(19)
    xyz = torch.rand(particles, 3, generator=generator)
    xyz[:, :2] = (xyz[:, :2]-0.5)*1.2
    xyz[:, 2] = 2.6+xyz[:, 2]*0.5
    colors = 0.2+0.7*torch.rand(particles, 3, generator=generator)
    scales = 0.07+0.05*torch.rand(particles, 3, generator=generator)
    q = torch.zeros(particles, 4)
    q[:, 0] = 1
    times = torch.linspace(0, 1, count)
    K = torch.tensor([[size*1.2, 0, (size-1)/2], [0, size*1.2, (size-1)/2], [0, 0, 1]])
    camera = Camera(K, torch.eye(4), size, size)
    renderer = ReferenceRenderer()

    def state_at(time):
        movement = torch.stack([0.12*torch.sin(2*math.pi*time),
                                0.05*torch.sin(math.pi*time), time*0])
        return GaussianState(xyz+movement, scales, q, colors, torch.full((particles, 1), 0.8))

    rgb, depth, valid, flow, flow_valid, next_indices = [], [], [], [], [], []
    with torch.no_grad():
        for index, time in enumerate(times):
            has_next = index < count-1
            r = renderer.render(state_at(time), camera, torch.full((particles, 1), 0.01),
                                state_at(times[index+1]) if has_next else None, camera)
            rgb.append(r.rgb.numpy())
            depth.append(r.depth.numpy())
            valid.append((r.alpha > 0.02).numpy())
            flow.append(r.flow.numpy() if has_next else np.zeros((size, size, 2), np.float32))
            flow_valid.append((r.alpha > 0.02).numpy() if has_next else np.zeros((size, size), bool))
            next_indices.append(index+1 if has_next else -1)
    test = np.arange(count) % 3 == 2
    np.savez_compressed(output / "frames.npz", rgb=np.array(rgb), depth=np.array(depth),
                        depth_valid=np.array(valid), flow=np.array(flow), flow_valid=np.array(flow_valid),
                        flow_next=np.array(next_indices, dtype=np.int64), K=K.expand(count, -1, -1).numpy(),
                        world_to_camera=np.tile(np.eye(4, dtype=np.float32), (count, 1, 1)),
                        times=times.numpy(), train=~test, test=test)
    np.savez_compressed(output / "canonical.npz", xyz=xyz.numpy(), rgb=colors.numpy(), scales=scales.numpy())
    metadata = {"schema_version": 1, "name": "synthetic-moving-gaussians", "dataset_kind": "synthetic",
                "protocol": "image_only", "geometry_source": "synthetic", "time_unit": "seconds",
                "length_unit": "normalized", "frames_file": "frames.npz", "canonical_file": "canonical.npz",
                "note": "Analytic toy sequence; reference-renderer self-consistency only, not benchmark evidence."}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output / "manifest.json"
