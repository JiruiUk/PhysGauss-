"""Official PIDG train -> render -> metrics, preserving released splits and configs."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import torch
from training_watchdog import run_watched, TrainingStalled

CONFIGS = {"mechanics_cloth": "mechanics_cloth", "balls_reaction": "balls_reaction",
           "dry_ice": "dry_ice", "motion_kuro": "motion_kuro", "rubber_duck": "rubber_dack"}
REPO_COMMIT = "b77fd2107a50c875d7219bb4f82388d5e5ea1319"


def command(arguments, log, cwd, append=False, allow_saved_stop=False):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", subprocess.list2cmdline(arguments), flush=True)
    with log.open("a" if append else "w", encoding="utf-8") as output:
        if allow_saved_stop:
            returncode = run_watched(arguments, output, log, cwd)
        else:
            returncode = subprocess.run(arguments, cwd=cwd, stdout=output, stderr=subprocess.STDOUT).returncode
    if allow_saved_stop and returncode == 75:
        return 75
    if returncode:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        print("\n".join(tail), flush=True)
        raise RuntimeError(f"Command failed ({returncode}); see {log}")
    return returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", choices=list(CONFIGS), default=["mechanics_cloth"])
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Load a full optimizer checkpoint in an existing run")
    parser.add_argument("--seed", type=int, default=0, help="Released code uses 0; paper states 42")
    parser.add_argument("--profile", choices=["official", "paper_hparams"], default="official")
    parser.add_argument("--tag", default="", help="Optional unique suffix; existing runs are preserved")
    args = parser.parse_args()
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("tag must contain only letters, numbers, underscores or hyphens")
    project = Path(__file__).resolve().parents[2]
    repo = project / "third_party/PIDG"
    entrypoint = Path(__file__).with_name("entrypoint.py")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official method")
    os.environ["PIDG_SEED"] = str(args.seed)
    if args.profile == "paper_hparams" and args.seed == 0:
        args.seed = 42
        os.environ["PIDG_SEED"] = "42"
    iterations = 120 if args.smoke else args.iterations
    run_root = project / "runs" / ("pidg_smoke" if args.smoke else f"pidg_{args.profile}")
    if args.tag:
        run_root = run_root.with_name(run_root.name + "_" + args.tag)
    for scene in args.scenes:
        if (run_root / scene).exists():
            if not args.resume or not (run_root / scene / "checkpoints/resume.pt").is_file():
                raise FileExistsError(f"Existing run requires --resume and a full checkpoint: {run_root / scene}")
    controller = {"run_root": str(run_root), "pid": os.getpid(), "started_at": time.time()}
    (project / "reproduction/controller.json").write_text(json.dumps(controller, indent=2), encoding="utf-8")
    for scene in args.scenes:
        dataset = project / "data/PIDG" / scene
        if not (dataset / "download_manifest.json").is_file() or not (dataset / "audit.json").is_file():
            raise RuntimeError(f"Download and audit {scene} first")
        run = run_root / scene
        resuming = run.exists()
        run.mkdir(parents=True, exist_ok=resuming)
        config = repo / "arguments/pidg" / (CONFIGS[scene] + ".py")
        metadata = {"scene": scene, "repo_commit": REPO_COMMIT,
                    "dataset_revision": "46b23520acaaf5d58938fb19b50a3a62d12c55df",
                    "profile": args.profile, "smoke_only": args.smoke, "seed": args.seed,
                    "iterations": iterations, "resolution_width": 400 if args.smoke else 1600,
                    "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(), "status": "training", "started_at": time.time(),
                    "note": "Patched official release on Windows. Paper/code discrepancies are documented; numeric agreement is not assumed."}
        status = run / "reproduction.json"
        if resuming:
            previous = json.loads(status.read_text(encoding="utf-8"))
            for key in ("scene", "profile", "seed", "iterations", "resolution_width", "repo_commit", "dataset_revision"):
                if previous[key] != metadata[key]:
                    raise ValueError(f"Cannot resume with a different {key}")
            if previous.get("status") == "completed":
                raise ValueError("Completed run must not be resumed")
            metadata = previous
            metadata["status"] = "training"
            metadata.pop("error", None)
            metadata.setdefault("resume_history", []).append(time.time())
        status.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        common = ["-s", str(dataset), "-m", str(run), "--conf", str(config),
                  "--load2gpu_on_the_fly", "--resolution", str(metadata["resolution_width"])]
        train = [sys.executable, "-u", str(entrypoint), "train_pidg.py", *common,
                 "--iterations", str(iterations), "--save_image_dir", str(run / "diagnostics")]
        if resuming:
            train += ["--resume_checkpoint", str(run / "checkpoints/resume.pt")]
        if args.smoke:
            train += ["--warm_up", "0", "--densify_until_iter", "0",
                      "--test_iterations", str(iterations), "--save_iterations", str(iterations)]
        if args.profile == "paper_hparams":
            train += ["--physics_loss_weight", "0.1"]
            if not args.smoke:
                train += ["--densify_until_iter", "30000"]
        try:
            for attempt in range(3):
                try:
                    result = command(train, run / "train.log", repo, append=resuming or attempt > 0, allow_saved_stop=True)
                    break
                except TrainingStalled as error:
                    metadata.setdefault("automatic_restarts", []).append({"unix": time.time(), "iteration": error.checkpoint["iteration"]})
                    status.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    if attempt == 2:
                        raise
                    if "--resume_checkpoint" not in train:
                        train += ["--resume_checkpoint", str(run / "checkpoints/resume.pt")]
                    print(f"WATCHDOG: restarting {scene} from full checkpoint {error.checkpoint['iteration']} ({attempt+1}/2)", flush=True)
            if result == 75:
                checkpoint = json.loads((run / "checkpoints/resume.json").read_text(encoding="utf-8"))
                metadata.update(status="saved_stopped", checkpoint_iteration=checkpoint["iteration"],
                                checkpoint_path=checkpoint["path"])
                status.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                print("Full checkpoint saved; remaining queue stopped by user request", flush=True)
                return
            metadata["status"] = "rendering"
            status.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            command([sys.executable, "-u", str(entrypoint), "render.py", *common,
                     "--iteration", "best", "--skip_train"], run / "render.log", repo)
            metadata["status"] = "evaluating"
            status.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            command([sys.executable, "-u", str(entrypoint), "metrics.py", "-m", str(run)],
                    run / "metrics.log", repo)
            results_path = run / "results.json"
            if not results_path.is_file() or not json.loads(results_path.read_text()):
                raise RuntimeError("Official evaluator did not produce nonempty results.json")
            metadata["status"] = "completed"
            metadata["metrics"] = json.loads(results_path.read_text())
            metadata["finished_at"] = time.time()
            status.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(json.dumps(metadata["metrics"], indent=2), flush=True)
        except Exception as error:
            metadata["status"] = "failed"
            metadata["error"] = str(error)
            status.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            raise


if __name__ == "__main__":
    main()
