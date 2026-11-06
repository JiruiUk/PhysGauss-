from pathlib import Path
import json
import math
import numpy as np
from PIL import Image
import torch
from ..config import TrainConfig, LossConfig
from ..losses import masked_mean, ssim_score
from ..rendering import ReferenceRenderer
from .trainer import create_model


def load_model(dataset, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["data_fingerprint"] != dataset.fingerprint():
        raise ValueError("Evaluation dataset differs from training checkpoint")
    raw = dict(checkpoint["config"])
    raw["loss"] = LossConfig(**raw["loss"])
    raw["device"] = device
    config = TrainConfig(**raw).validate()
    torch.set_num_threads(config.cpu_threads)
    model = create_model(dataset, config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate(dataset, checkpoint_path, output, device="cpu"):
    model, checkpoint = load_model(dataset, checkpoint_path, device)
    renderer = ReferenceRenderer()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index in dataset.test_indices:
        frame = dataset.frame(index)
        state, variance = model(frame.time)
        next_state = model(frame.next_time)[0] if frame.next_time is not None else None
        result = renderer.render(state, frame.camera, variance, next_state, frame.next_camera)
        mse = float(masked_mean((result.rgb-frame.rgb).square(), frame.rgb_valid))
        record = {"frame": index, "psnr": -10*math.log10(max(mse, 1e-12))}
        if bool(frame.rgb_valid.all()):
            record["ssim_reference"] = float(ssim_score(result.rgb, frame.rgb))
        if frame.depth is not None and bool(frame.depth_valid.any()):
            record["depth_mae"] = float(masked_mean((result.depth-frame.depth).abs(), frame.depth_valid))
        if frame.flow is not None and bool(frame.flow_valid.any()):
            record["flow_epe"] = float(masked_mean(torch.linalg.vector_norm(result.flow-frame.flow, dim=-1), frame.flow_valid))
        records.append(record)
        comparison = torch.cat([frame.rgb, result.rgb, (result.rgb-frame.rgb).abs()], dim=1)
        Image.fromarray((comparison.clamp(0, 1).cpu().numpy()*255).astype(np.uint8)).save(output/f"frame_{index:04d}.png")
    keys = set().union(*(record.keys() for record in records)) - {"frame"}
    summary = {key: sum(r[key] for r in records if key in r)/sum(key in r for r in records) for key in keys}
    report = {"split": "test", "trained_steps": checkpoint["step"], "protocol": dataset.metadata["protocol"],
              "dataset_kind": dataset.metadata["dataset_kind"], "renderer": "torch_reference",
              "benchmark_compatible": False, "lpips": None,
              "limitations": "Toy/reference validation. SSIM uses a local 3x3 box; LPIPS not implemented. No physical-parameter ground truth evaluation.",
              "summary": summary, "frames": records}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return report


@torch.no_grad()
def export_state(dataset, checkpoint_path, time, output, device="cpu"):
    model, checkpoint = load_model(dataset, checkpoint_path, device)
    state, variance = model(time)
    velocity, stress = model.material(state.means, time)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"means": state.means, "scales": state.scales, "quaternions": state.quaternions,
              "rgb": state.colors, "opacity": state.opacities, "velocity": velocity,
              "cauchy_stress": stress, "photometric_variance": variance}
    np.savez_compressed(output, **{k: v.cpu().numpy() for k, v in arrays.items()},
                        time_seconds=np.array(time), trained_steps=np.array(checkpoint["step"]))
    return output
