"""Run isolated PIDG A0 and a paired original-source control sequentially."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
from pidg_benchmark import (PROJECT, SCENES, METRICS, compare_runs, completed_metrics,
                            dataset_hashes, digest_mapping, runtime_hashes, sha256,
                            write_json)
from check_pidg_baseline import audit


def execute(command, log, repo, metadata, status):
    metadata.setdefault("commands", []).append(command)
    write_json(status, metadata)
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    with log.open("w", encoding="utf-8") as output:
        code = subprocess.run(command, cwd=repo, stdin=subprocess.DEVNULL,
                              stdout=output, stderr=subprocess.STDOUT).returncode
    if code == 75:
        metadata["status"] = "saved_stopped"
        write_json(status, metadata)
        raise SystemExit(75)
    if code:
        raise RuntimeError(f"Command failed ({code}); see {log}")


def run_one(repo, run, scene, smoke, seed, iterations, torch):
    source_hashes = runtime_hashes(repo)
    data_hashes = dataset_hashes(scene)
    config = repo / "arguments/pidg" / (SCENES[scene] + ".py")
    scene_options = runpy.run_path(str(config))
    import diff_gaussian_rasterization
    import diff_gaussian_rasterization._C as rasterizer_binary
    import simple_knn._C as knn_binary
    shared = [PROJECT / "reproduction/scripts/training_checkpoint.py",
              Path(diff_gaussian_rasterization.__file__), Path(rasterizer_binary.__file__),
              Path(knn_binary.__file__)]
    protocol = {"scene": scene, "profile": "official", "seed": seed,
                "iterations": iterations, "resolution_width": 400 if smoke else 1600,
                "smoke_only": smoke, "warm_up": scene_options.get("warm_up", 0 if smoke else 3000),
                "densify_until_iter": 0 if smoke else 15000,
                "physics_loss_weight": 0.01, "flow_loss_weight": 0.01,
                "dataset_hashes": data_hashes, "config_sha256": sha256(config),
                "evaluator_sha256": sha256(repo / "metrics.py"),
                "render_sha256": sha256(repo / "render.py"),
                "shared_runtime_hashes": {str(p): sha256(p) for p in shared},
                "snapshot": "best_test_psnr", "device": torch.cuda.get_device_name(),
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "python": sys.version}
    run.mkdir(parents=True, exist_ok=False)
    metadata = {**{k: protocol[k] for k in ("scene", "profile", "seed", "iterations",
                                          "resolution_width", "smoke_only")},
                "protocol": protocol, "source_root": str(repo),
                "source_digest": digest_mapping(source_hashes), "status": "training",
                "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(run / "source_hashes.json", source_hashes)
    status = run / "reproduction.json"
    entry = PROJECT / "scripts/pidg_variant_entrypoint.py"
    prefix = [sys.executable, "-u", str(entry), "--repo", str(repo), "--seed", str(seed)]
    dataset = PROJECT / "data/PIDG" / scene
    common = ["-s", str(dataset), "-m", str(run), "--conf", str(config),
              "--load2gpu_on_the_fly", "--resolution", str(protocol["resolution_width"])]
    train = [*prefix, "train_pidg.py", *common, "--iterations", str(iterations),
             "--save_image_dir", str(run / "diagnostics"),
             "--checkpoint_interval", "120" if smoke else "1000"]
    if smoke:
        train += ["--warm_up", "0", "--densify_until_iter", "0", "--test_iterations",
                  str(iterations), "--save_iterations", str(iterations)]
    try:
        execute(train, run / "train.log", repo, metadata, status)
        metadata["status"] = "rendering"
        execute([*prefix, "render.py", *common, "--iteration", "best", "--skip_train"],
                run / "render.log", repo, metadata, status)
        metadata["status"] = "evaluating"
        execute([*prefix, "metrics.py", "-m", str(run)], run / "metrics.log", repo, metadata, status)
        if runtime_hashes(repo) != source_hashes or dataset_hashes(scene) != data_hashes:
            raise RuntimeError("Source or data changed during the run")
        metadata.update(status="completed", metrics=json.loads((run / "results.json").read_text()),
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(status, metadata)
        completed_metrics(run)  # Official evaluator catches errors internally: validate its output.
        log = (run / "train.log").read_text(encoding="utf-8", errors="replace")
        if smoke and "Training complete." not in log:
            raise RuntimeError("Sanity run did not finish training")
        if smoke and scene_options.get("use_flow", False) and "100 Optical flow loss" not in log:
            raise RuntimeError("Sanity run did not exercise configured flow supervision")
        if smoke:
            sys.path.append(str(PROJECT / "reproduction/scripts"))
            from training_watchdog import verified_checkpoint
            if verified_checkpoint(run / "checkpoints")["iteration"] != iterations:
                raise RuntimeError("Sanity checkpoint iteration differs")
        print(json.dumps(metadata["metrics"], indent=2), flush=True)
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        write_json(status, metadata)
        raise


def write_comparison(root, scenes, smoke):
    pairs = [compare_runs(root / "control" / scene, root / "candidate" / scene) for scene in scenes]
    report = {"phase": 0, "mode": "smoke" if smoke else "benchmark", "pairs": pairs,
              "all_within_tolerance": all(p["within_tolerance"] for p in pairs),
              "historical_formal_reference": audit(PROJECT / "variants/phase00_baseline")["formal_baseline"]}
    write_json(root / "comparison.json", report)
    text = "# Phase 0 paired PIDG comparison\n\n"
    text += ("120-step, width-400 engineering sanity; not a 50k research benchmark.\n\n" if smoke
             else "50k, width-1600 official protocol.\n\n")
    text += "Delta is candidate minus original-source control. Lower LPIPS is better.\n\n"
    text += "| Scene | Metric | Control | Isolated A0 | Delta |\n|---|---|---:|---:|---:|\n"
    for pair in pairs:
        for key in METRICS:
            text += (f"| {pair['scene']} | {key} | {pair['control_metrics'][key]:.6f} | "
                     f"{pair['candidate_metrics'][key]:.6f} | {pair['delta_candidate_minus_control'][key]:+.6f} |\n")
    text += f"\nWithin predeclared engineering tolerances: {report['all_within_tolerance']}.\n"
    text += "\n## Previous 50k benchmark (reference only)\n\n"
    text += "| Scene | PSNR | SSIM | LPIPS |\n|---|---:|---:|---:|\n"
    for row in report["historical_formal_reference"]:
        m = row["metrics"]
        text += f"| {row['scene']} | {m['PSNR']:.5f} | {m['SSIM']:.5f} | {m['LPIPS']:.5f} |\n"
    text += "\nNo cross-budget gains are computed. Historical code hashes were not recorded at training time.\n"
    (root / "comparison.md").write_text(text, encoding="utf-8")
    if not report["all_within_tolerance"]:
        raise RuntimeError("A0 parity gate failed; inspect comparison.json before proceeding")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "benchmark"), default="smoke")
    parser.add_argument("--scenes", nargs="+", choices=list(SCENES), default=["mechanics_cloth"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("Invalid tag")
    if len(args.scenes) != len(set(args.scenes)):
        raise ValueError("Duplicate scenes")
    variant = PROJECT / "variants/phase00_baseline"
    baseline_audit = audit(variant)
    root = variant / "runs" / args.tag
    if root.exists():
        raise FileExistsError(f"Preserving existing run; choose a new tag: {root}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Official PIDG requires CUDA")
    root.mkdir(parents=True)
    write_json(root / "baseline_audit.json", baseline_audit)
    smoke = args.mode == "smoke"
    for scene in args.scenes:
        for label, repo in (("control", PROJECT / "third_party/PIDG"), ("candidate", variant / "code")):
            run_one(repo, root / label / scene, scene, smoke, args.seed, 120 if smoke else 50000, torch)
    write_comparison(root, args.scenes, smoke)
    print(f"PASS: {root / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
