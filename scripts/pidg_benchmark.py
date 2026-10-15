"""Local PIDG provenance and protocol checks (no CUDA imports)."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
SCENES = {"mechanics_cloth": "mechanics_cloth", "balls_reaction": "balls_reaction",
          "dry_ice": "dry_ice", "motion_kuro": "motion_kuro", "rubber_duck": "rubber_dack"}
METRICS = ("PSNR", "SSIM", "LPIPS", "MS-SSIM")
# Predeclared engineering tolerances, not confidence intervals or research gains.
TOLERANCES = {"PSNR": 0.1, "SSIM": 0.005, "LPIPS": 0.005, "MS-SSIM": 0.005}
RUNTIME_DIRS = ("arguments", "gaussian_renderer", "hashencoder", "lpipsPyTorch",
                "motion_utils", "scene_PIDG", "utils", "physics", "models", "losses")
RUNTIME_FILES = ("train_pidg.py", "render.py", "metrics.py")


def read_json(path):
    for attempt in range(20):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except (PermissionError, json.JSONDecodeError):
            if attempt==19: raise
            time.sleep(.01)


def write_text_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value,encoding='utf-8')
        for attempt in range(20):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt==19: raise
                time.sleep(.01)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, data):
    write_text_atomic(path,json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_hashes(repo):
    repo = Path(repo)
    paths = [repo / name for name in RUNTIME_FILES]
    if (repo / "training_checkpoint.py").is_file():
        paths.append(repo / "training_checkpoint.py")
    for name in RUNTIME_DIRS:
        paths.extend(p for p in (repo / name).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".cu", ".cpp", ".h", ".cuh"}
                     and "__pycache__" not in p.parts)
    return {p.relative_to(repo).as_posix(): sha256(p) for p in sorted(paths)}


def digest_mapping(mapping):
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def dataset_hashes(scene):
    root = PROJECT / "data/PIDG" / scene
    names = ("transforms_train.json", "transforms_test.json", "points3d.ply",
             "download_manifest.json", "audit.json")
    # points3d.ply is mandatory: do not regenerate random canonical points.
    return {name: sha256(root / name) for name in names}


def completed_metrics(run):
    run = Path(run)
    metadata = read_json(run / "reproduction.json")
    if metadata.get("status") != "completed":
        raise ValueError(f"Incomplete run: {run}")
    result = read_json(run / "results.json")["ours_best"]
    for key in METRICS:
        value = result[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Nonfinite or missing {key}: {run}")
    if metadata.get("metrics", {}).get("ours_best") != result:
        raise ValueError(f"Metadata/results mismatch: {run}")
    return metadata, result


def compare_runs(control, candidate):
    left, a = completed_metrics(control)
    right, b = completed_metrics(candidate)
    if left["protocol"] != right["protocol"]:
        raise ValueError("Cannot compare runs with different protocols")
    delta = {key: b[key] - a[key] for key in METRICS}
    return {"scene": right["scene"], "control": str(control), "candidate": str(candidate),
            "protocol": right["protocol"], "control_metrics": a, "candidate_metrics": b,
            "delta_candidate_minus_control": delta, "tolerances": TOLERANCES,
            "within_tolerance": all(abs(delta[k]) <= TOLERANCES[k] for k in METRICS)}
