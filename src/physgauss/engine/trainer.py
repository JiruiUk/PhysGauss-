from pathlib import Path
import json
import random
import torch
from ..models import PhysGaussScene
from ..rendering import ReferenceRenderer
from ..losses import visual_losses
from ..physics import momentum_residual, trajectory_residual
from ..physics.adaptive import AdaptiveCollocation, physics_scale


def seed_everything(seed, cpu_threads):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)


def create_model(dataset, config):
    return PhysGaussScene(dataset.xyz, dataset.colors, dataset.scales, config).to(config.device)


def train(dataset, config, output, resume=None):
    config.validate()
    seed_everything(config.seed, config.cpu_threads)
    output = Path(output)
    if not resume and (output / "checkpoint.pt").exists():
        raise FileExistsError("Existing run: use --resume or choose another output directory")
    output.mkdir(parents=True, exist_ok=True)
    model = create_model(dataset, config)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.lr)
    renderer = ReferenceRenderer()
    sampler = AdaptiveCollocation(len(dataset.xyz), config.device, config.adaptive_uniform_fraction)
    fingerprint = dataset.fingerprint()
    start = 0
    if resume:
        checkpoint = torch.load(resume, map_location=config.device, weights_only=True)
        if checkpoint["data_fingerprint"] != fingerprint:
            raise ValueError("Resume dataset differs from checkpoint")
        mutable = {"steps", "device", "cpu_threads", "log_every", "checkpoint_every"}
        current = {k: v for k, v in config.to_dict().items() if k not in mutable}
        previous = {k: v for k, v in checkpoint["config"].items() if k not in mutable}
        if current != previous:
            raise ValueError("Resume requires the same model, optimizer, loss and physics settings")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        sampler.scores.copy_(checkpoint["collocation_scores"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if config.device.startswith("cuda") and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
        start = checkpoint["step"]
        if config.steps <= start:
            raise ValueError("--steps is the total target; it must exceed the resumed step")
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (output / "data_provenance.json").write_text(json.dumps({
        "manifest": str(dataset.path), "sha256": fingerprint, "metadata": dataset.metadata,
        "train_frames": dataset.train_indices, "test_frames": dataset.test_indices,
        "renderer": "torch_reference", "benchmark_compatible": False,
    }, indent=2), encoding="utf-8")

    def save(step):
        checkpoint = {"format_version": 1, "step": step, "config": config.to_dict(),
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "collocation_scores": sampler.scores, "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if config.device.startswith("cuda") else [],
                      "data_fingerprint": fingerprint, "data_metadata": dataset.metadata}
        temporary = output / "checkpoint.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "checkpoint.pt")

    with (output / "train.jsonl").open("a" if resume else "w", encoding="utf-8") as log:
        for step in range(start, config.steps):
            model.train()
            frame_index = dataset.train_indices[step % len(dataset.train_indices)]
            frame = dataset.frame(frame_index, training=True)
            state, variance = model(frame.time)
            next_state = model(frame.next_time)[0] if frame.next_time is not None else None
            rendered = renderer.render(state, frame.camera, variance, next_state, frame.next_camera)
            losses = visual_losses(rendered, frame)
            factor = physics_scale(step+1, config.physics_warmup, config.physics_ramp)
            losses["trajectory"] = rendered.rgb.sum()*0
            losses["momentum"] = rendered.rgb.sum()*0
            if factor > 0 and (config.loss.trajectory > 0 or config.loss.momentum > 0):
                if config.adaptive_sampling:
                    indices = sampler.sample(config.collocation_count)
                else:
                    indices = torch.randperm(len(dataset.xyz), device=config.device)[:config.collocation_count]
                if config.loss.trajectory > 0:
                    residual = trajectory_residual(model.deformation, model.material,
                                                   model.canonical.means[indices], frame.time)
                    losses["trajectory"] = residual.square().mean()
                if config.loss.momentum > 0:
                    residual = momentum_residual(model.material, state.means[indices], frame.time,
                                                 config.density, config.body_acceleration)
                    losses["momentum"] = residual.square().mean()
                    if config.adaptive_sampling:
                        sampler.update(indices, residual)
            total = sum(getattr(config.loss, key)*value*(factor if key in {"trajectory", "momentum"} else 1)
                        for key, value in losses.items())
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite loss at step {step+1}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            record = {"step": step+1, "frame": frame_index, "total": float(total.detach()),
                      "physics_scale": factor, "gradient_norm": float(gradient_norm),
                      **{key: float(value.detach()) for key, value in losses.items()}}
            log.write(json.dumps(record)+"\n")
            log.flush()
            if step == start or (step+1) % config.log_every == 0 or step+1 == config.steps:
                print(f"step={step+1:04d} total={record['total']:.6f} rgb={record['rgb']:.6f} physics={factor:.2f}", flush=True)
            if (step+1) % config.checkpoint_every == 0:
                save(step+1)
    save(config.steps)
    return output / "checkpoint.pt"
