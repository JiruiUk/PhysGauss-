"""Resumable, sequential official-protocol long training of frozen PIDG variants."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from pidg_benchmark import (PROJECT, SCENES, METRICS, read_json, write_json, sha256,
                            runtime_hashes, dataset_hashes, completed_metrics, write_text_atomic)
from snapshot_upgrade import NAMES
from upgrade_scope import selected_scenes

QUEUE = PROJECT / "variants/upgrade"


class UpgradeStalled(RuntimeError):
    def __init__(self,checkpoint):
        self.checkpoint=checkpoint
        super().__init__(f"Stalled child; verified checkpoint {checkpoint['iteration']}")


class QueueLock:
    def __enter__(self):
        self.stream=(QUEUE/'controller.lock').open('a+b')
        self.stream.seek(0)
        if not self.stream.read(1): self.stream.write(b'0');self.stream.flush()
        self.stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(self.stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return self

    def __exit__(self,*args): self.stream.close()


def run_command(command, repo, run, step, env):
    sys.path.append(str(PROJECT / "reproduction/scripts"))
    from training_watchdog import verified_checkpoint
    log = run / (step + ".log")
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\nCOMMAND " + subprocess.list2cmdline(command) + "\n")
        stream.flush()
        child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT)
        write_json(QUEUE / "active.json", {"controller_pid": os.getpid(), "child_pid": child.pid,
                    "step": step, "run": str(run), "started_utc": datetime.now(timezone.utc).isoformat()})
        previous = log.stat().st_mtime_ns
        last_activity = time.monotonic()
        last_report=last_activity
        stop_forwarded=False
        while child.poll() is None:
            if time.monotonic()-last_report>30:
                report()
                last_report=time.monotonic()
            if (QUEUE / "stop_queue.request").exists() and step == "train" and not stop_forwarded:
                (run / "checkpoints").mkdir(exist_ok=True)
                (run / "checkpoints/save_and_stop.request").write_text("save and exit", encoding="utf-8")
                stop_forwarded=True
            stamp = log.stat().st_mtime_ns
            if stamp != previous:
                previous, last_activity = stamp, time.monotonic()
            if step == "train" and time.monotonic() - last_activity > 900:
                if (QUEUE/'stop_queue.request').exists():
                    # A requested end-of-iteration save is pending; preserve the live process.
                    time.sleep(1)
                    continue
                # Terminate only this exact child, after validating a full disk checkpoint.
                saved = verified_checkpoint(run / "checkpoints")
                child.kill()
                child.wait()
                raise UpgradeStalled(saved)
            time.sleep(1)
    return child.returncode


def run_scene(variant, scene, mode):
    import torch
    stage_info=read_json(variant / "stage.json")
    repo = variant / stage_info.get("code_dir","code")
    expected = read_json(variant / "source_manifest.json")["files"]
    if runtime_hashes(repo) != expected:
        raise ValueError(f"Frozen source changed: {variant}")
    revision=stage_info.get("revision",0)
    run_mode=mode if mode!="smoke" or revision==0 else f"smoke_r{revision}"
    run = variant / "runs" / run_mode / scene
    config = read_json(variant / "config.json")
    dataset = PROJECT / "data/PIDG" / scene
    smoke = mode == "smoke"
    protocol = {"scene": scene, "iterations": 120 if smoke else 50000,
                "resolution_width": 400 if smoke else 1600, "seed": 0, "profile": "official",
                "dataset_hashes": dataset_hashes(scene), "config_sha256": sha256(variant / "config.json"),
                "scene_config_sha256": sha256(repo / f"arguments/pidg/{SCENES[scene]}.py"),
                "evaluator_sha256": sha256(repo / "metrics.py"), "torch": str(torch.__version__),
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}
    status = run / "reproduction.json"
    resuming = False
    if status.exists():
        metadata = read_json(status)
        if metadata["protocol"] != protocol or metadata["source_hashes"] != expected:
            raise ValueError("Cannot resume with changed protocol/source")
        if metadata["status"] == "completed":
            completed_metrics(run)
            return
        checkpoint = run / "checkpoints/resume.pt"
        if checkpoint.exists():
            sys.path.append(str(PROJECT / "reproduction/scripts"))
            from training_watchdog import verified_checkpoint
            verified_checkpoint(checkpoint.parent)
            resuming = True
            pending=checkpoint.parent/'save_and_stop.request'
            if metadata['status']=='saved_stopped' and pending.exists() and not (QUEUE/'stop_queue.request').exists():
                pending.rename(pending.with_name(f'save_and_stop.acknowledged_{time.time_ns()}'))
        elif metadata["status"] not in {"rendering", "evaluating"}:
            raise RuntimeError(f"No full checkpoint; preserve failed run and diagnose: {run}")
    else:
        run.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(variant / "config.json", run / "physics_config.json")
        metadata = {"scene": scene, "phase": read_json(variant / "stage.json")["phase"],
                    "protocol": protocol, "source_hashes": expected, "iterations": protocol["iterations"],
                    "resolution_width": protocol["resolution_width"], "seed": 0, "smoke_only": smoke,
                    "profile": "official", "status": "training", "started_utc": datetime.now(timezone.utc).isoformat()}
        import diff_gaussian_rasterization._C as rasterizer
        import simple_knn._C as knn
        helper=repo/'training_checkpoint.py'
        if not helper.exists(): helper=PROJECT/'reproduction/scripts/training_checkpoint.py'
        metadata['shared_runtime_hashes']={str(p):sha256(p) for p in (Path(rasterizer.__file__),Path(knn.__file__),helper)}
        write_json(status, metadata)
    env = os.environ.copy()
    env["PIDG_UPGRADE_CONFIG"] = str(run / "physics_config.json")
    if smoke:
        env["PIDG_UPGRADE_SMOKE"] = "1"
    else:
        env.pop("PIDG_UPGRADE_SMOKE",None)
    env["PIDG_HASH_SOURCE_DIR"] = str(PROJECT / "third_party/PIDG/hashencoder")
    env["PIDG_HASH_BUILD_DIR"] = str(PROJECT / "third_party/PIDG/tmp_build")
    # Shared source must be identical to the frozen variant's encoder kernels.
    for file in (repo / "hashencoder/src").glob("*"):
        if file.is_file() and sha256(file) != sha256(PROJECT / "third_party/PIDG/hashencoder/src" / file.name):
            raise ValueError("Shared hash CUDA kernel source differs")
    prefix = [sys.executable, "-u", str(PROJECT / "scripts/pidg_variant_entrypoint.py"),
              "--repo", str(repo), "--seed", "0"]
    common = ["-s", str(dataset), "-m", str(run), "--conf", str(repo / f"arguments/pidg/{SCENES[scene]}.py"),
              "--load2gpu_on_the_fly", "--resolution", str(protocol["resolution_width"])]
    command = [*prefix, "train_pidg.py", *common, "--iterations", str(protocol["iterations"]),
               "--save_image_dir", str(run / "diagnostics"), "--checkpoint_interval", "120" if smoke else "1000"]
    if smoke:
        command += ["--warm_up", "0", "--densify_until_iter", "0", "--test_iterations", "120", "--save_iterations", "120"]
    if resuming:
        command += ["--resume_checkpoint", str(run / "checkpoints/resume.pt")]
    try:
        if metadata["status"] not in {"rendering", "evaluating"}:
            metadata["status"] = "training"
            write_json(status, metadata)
            for attempt in range(3):
                try:
                    code = run_command(command, repo, run, "train", env)
                    break
                except UpgradeStalled as error:
                    if attempt==2: raise
                    metadata.setdefault('automatic_restarts',[]).append({'iteration':error.checkpoint['iteration'],
                                           'sha256':error.checkpoint['sha256'],'attempt':attempt+1})
                    write_json(status,metadata)
                    if '--resume_checkpoint' not in command:
                        command+=['--resume_checkpoint',str(run/'checkpoints/resume.pt')]
            if code == 75:
                metadata["status"] = "saved_stopped"
                write_json(status, metadata)
                return 75
            if code:
                raise RuntimeError(f"Training failed ({code}): {run / 'train.log'}")
        metadata["status"] = "rendering"
        write_json(status, metadata)
        if run_command([*prefix, "render.py", *common, "--iteration", "best", "--skip_train"], repo, run, "render", env):
            raise RuntimeError("Rendering failed")
        metadata["status"] = "evaluating"
        write_json(status, metadata)
        if run_command([*prefix, "metrics.py", "-m", str(run)], repo, run, "metrics", env):
            raise RuntimeError("Evaluation failed")
        if runtime_hashes(repo) != expected or dataset_hashes(scene) != protocol["dataset_hashes"]:
            raise RuntimeError("Source/data changed during training")
        metadata.update(status="completed", metrics=read_json(run / "results.json"),
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(status, metadata)
        completed_metrics(run)
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        write_json(status, metadata)
        raise


def report():
    scenes = selected_scenes(QUEUE)
    rows = []
    for phase, name in NAMES.items():
        variant = PROJECT / f"variants/phase{phase:02d}_{name}"
        for scene in scenes:
            run = variant / "runs/long" / scene
            row = {"phase": phase, "scene": scene, "status": "not_implemented" if not variant.exists() else "queued"}
            if (run / "reproduction.json").exists():
                info = read_json(run / "reproduction.json")
                row["status"] = info["status"]
                if (run / "checkpoints/resume.json").exists():
                    row["checkpoint_iteration"] = read_json(run / "checkpoints/resume.json")["iteration"]
                if info["status"] == "completed":
                    _, metrics = completed_metrics(run)
                    baseline, reference = completed_metrics(PROJECT / "runs/pidg_official" / scene)
                    for key in ("iterations", "resolution_width", "seed", "profile"):
                        if info[key] != baseline[key]:
                            raise ValueError(f"Historical protocol mismatch: {key}")
                    row["metrics"] = metrics
                    row["baseline"] = reference
                    row["delta"] = {key: metrics[key]-reference[key] for key in METRICS}
            rows.append(row)
    write_json(QUEUE / "results.json", {"scenes": scenes, "excluded_scenes": [s for s in SCENES if s not in scenes],
               "rows": rows, "all_completed": all(r["status"]=="completed" for r in rows)})
    text = "# PIDG upgrade long-training benchmark\n\n"
    text += f"Selected scenes per phase: {', '.join(scenes)}; {len(rows)} runs; 50,000 iterations, width 1600, seed 0.\n\n"
    text += "Completion applies to the selected scenes only. Other scenes are outside the current training scope.\n\n"
    text += "| Phase | Scene | Status | Checkpoint | PSNR | Delta PSNR | SSIM | LPIPS |\n|---|---|---|---:|---:|---:|---:|---:|\n"
    for row in rows:
        m, d = row.get("metrics", {}), row.get("delta", {})
        def fmt(values, key): return f"{values[key]:.5f}" if key in values else "—"
        text += f"| {row['phase']} | {row['scene']} | {row['status']} | {row.get('checkpoint_iteration', '—')} | {fmt(m,'PSNR')} | {fmt(d,'PSNR')} | {fmt(m,'SSIM')} | {fmt(m,'LPIPS')} |\n"
    text += "\nOnly completed long runs have metric deltas. Historical baselines lack runtime source hashes; current inputs and evaluator are pinned for new runs.\n"
    write_text_atomic(QUEUE / "results.md",text)


def verify_smoke(variant):
    stage=read_json(variant/'stage.json')
    revision=stage.get('revision',0)
    mode='smoke' if revision==0 else f'smoke_r{revision}'
    run=variant/'runs'/mode/'mechanics_cloth'
    completed_metrics(run)
    sys.path.append(str(PROJECT/'reproduction/scripts'))
    from training_watchdog import verified_checkpoint
    assert verified_checkpoint(run/'checkpoints')['iteration']==120
    if stage['phase']>=2:
        records=[json.loads(line) for line in (run/'physics.jsonl').read_text(encoding='utf-8').splitlines()]
        record=records[-1]
        required=['loss_kin','mean_speed']
        if stage['phase']>=3: required+=['mean_acceleration']
        if stage['phase']>=4: required+=['mean_detF','min_detF','max_detF']
        if stage['phase']>=5: required+=['loss_mass']
        if stage['phase']>=6: required+=['mean_stress_norm']
        if stage['phase']>=8: required+=['loss_momentum']
        if stage['phase']>=13: required+=['flow_valid_fraction','flow_epe_pixels']
        if stage['phase']>=15: required+=['loss_rollout']
        if any(record.get(key) is None for key in required):
            raise RuntimeError(f'Phase {stage["phase"]} did not exercise required physics: {required}')
        if stage['phase']>=4 and record['nonpositive_detF_count']/record['quadrature_count']>.05:
            raise RuntimeError('Short-run Jacobian gate failed; preserve logs and stabilize before long training')
        write_json(variant/'reports/smoke_physics.json',record)


def execute_queue(args):
    if (QUEUE/'stop_queue.request').exists():
        raise RuntimeError('Queue stop request present; acknowledge it before resuming')
    args.scenes = args.scenes if args.scenes is not None else selected_scenes(QUEUE)
    write_json(QUEUE/'benchmark_scope.json', {'scenes':args.scenes,
               'updated_utc':datetime.now(timezone.utc).isoformat()})
    controller={'pid':os.getpid(),'stages':args.stages,'scenes':args.scenes,'mode':args.mode,'status':'running'}
    write_json(QUEUE/'controller.json',controller)
    stage=None
    variant=None
    try:
        for phase in args.stages:
            if (QUEUE/'stop_queue.request').exists():
                controller['status']='saved_stopped'
                return
            variant=PROJECT/f'variants/phase{phase:02d}_{NAMES[phase]}'
            stage=read_json(variant/'stage.json')
            if stage['unit_tests']!='passed': raise RuntimeError(f'Phase {phase} unit gate pending')
            if args.mode in {'smoke','all'}:
                if run_scene(variant,'mechanics_cloth','smoke')==75:
                    controller['status']='saved_stopped'
                    return
                verify_smoke(variant)
                stage['smoke']='completed'
                write_json(variant/'stage.json',stage)
            if args.mode in {'long','all'}:
                if stage['smoke']!='completed': raise RuntimeError('Complete smoke gate first')
                stage['long_training']='running'
                stage['long_training_scenes']=list(args.scenes)
                write_json(variant/'stage.json',stage)
                for scene in args.scenes:
                    if shutil.disk_usage(PROJECT).free<15*1024**3: raise RuntimeError('Less than 15 GiB free')
                    report()
                    if (QUEUE/'stop_queue.request').exists():
                        controller['status']='saved_stopped'
                        return
                    if run_scene(variant,scene,'long')==75:
                        controller['status']='saved_stopped'
                        return
                    report()
                all_done=all((variant/f'runs/long/{scene}/reproduction.json').exists() and
                   read_json(variant/f'runs/long/{scene}/reproduction.json')['status']=='completed' for scene in args.scenes)
                if all_done:
                    stage.update(long_training='completed',status='long_trained',acceptance='pending_metric_review')
                    write_json(variant/'stage.json',stage)
        controller['status']='completed'
    except Exception as error:
        controller.update(status='failed',error=str(error))
        if stage is not None:
            stage.update(status='requires_attention',error=str(error))
            write_json(variant/'stage.json',stage)
        raise
    finally:
        controller['updated_utc']=datetime.now(timezone.utc).isoformat()
        write_json(QUEUE/'controller.json',controller)
        report()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stages',nargs='+',type=int,choices=range(1,16),default=list(range(1,16)))
    parser.add_argument('--mode',choices=('smoke','long','all'),default='all')
    parser.add_argument('--scenes',nargs='+',choices=list(SCENES),default=None,
                        help='Defaults to persisted benchmark_scope.json; all five if no scope exists')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.scenes is None: args.scenes=selected_scenes(QUEUE)
    QUEUE.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        report()
        return
    if len(set(args.stages))!=len(args.stages) or len(set(args.scenes))!=len(args.scenes):
        raise ValueError('Duplicate stages/scenes')
    with QueueLock(): execute_queue(args)


if __name__=='__main__': main()
