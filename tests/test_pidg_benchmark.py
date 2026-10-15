"""Guard fair official-PIDG comparisons and isolated runtime provenance."""
import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pidg_benchmark import METRICS, compare_runs, completed_metrics, runtime_hashes, write_json
from check_pidg_baseline import audit


def make_run(root, protocol=None, psnr=20.0):
    metrics = {"PSNR": psnr, "SSIM": 0.8, "LPIPS": 0.2, "MS-SSIM": 0.75}
    info = {"status": "completed", "scene": "mechanics_cloth",
            "protocol": copy.deepcopy(protocol or {"iterations": 120, "seed": 0}),
            "metrics": {"ours_best": metrics}}
    write_json(root / "reproduction.json", info)
    write_json(root / "results.json", info["metrics"])
    return info


def test_paired_comparison_and_tolerance_failure(tmp_path):
    control, candidate = tmp_path / "control", tmp_path / "candidate"
    make_run(control)
    make_run(candidate, psnr=20.03)
    result = compare_runs(control, candidate)
    assert result["within_tolerance"]
    assert result["delta_candidate_minus_control"]["PSNR"] == pytest.approx(0.03)
    make_run(candidate, psnr=20.3)
    assert not compare_runs(control, candidate)["within_tolerance"]


@pytest.mark.parametrize("key,value", [("iterations", 50000), ("seed", 42),
                                      ("config_sha256", "different"), ("dataset_hashes", {"split": "other"}),
                                      ("evaluator_sha256", "other")])
def test_protocol_mismatch_is_rejected(tmp_path, key, value):
    control, candidate = tmp_path / "control", tmp_path / "candidate"
    make_run(control)
    protocol = {"iterations": 120, "seed": 0, key: value}
    make_run(candidate, protocol=protocol)
    with pytest.raises(ValueError, match="different protocols"):
        compare_runs(control, candidate)


def test_incomplete_and_inconsistent_results_rejected(tmp_path):
    info = make_run(tmp_path)
    info["status"] = "training"
    write_json(tmp_path / "reproduction.json", info)
    with pytest.raises(ValueError, match="Incomplete"):
        completed_metrics(tmp_path)
    info["status"] = "completed"
    info["metrics"]["ours_best"]["PSNR"] = 12
    write_json(tmp_path / "reproduction.json", info)
    with pytest.raises(ValueError, match="mismatch"):
        completed_metrics(tmp_path)


@pytest.mark.parametrize("metric", METRICS)
def test_nonfinite_metrics_are_rejected(tmp_path, metric):
    info = make_run(tmp_path)
    info["metrics"]["ours_best"][metric] = float("nan")
    write_json(tmp_path / "reproduction.json", info)
    write_json(tmp_path / "results.json", info["metrics"])
    with pytest.raises(ValueError, match="Nonfinite"):
        completed_metrics(tmp_path)


def test_runtime_fingerprint_detects_changes_and_ignores_cache(tmp_path):
    for name in ("train_pidg.py", "render.py", "metrics.py"):
        (tmp_path / name).write_text("# original\n")
    original = runtime_hashes(tmp_path)
    (tmp_path / "tmp_build").mkdir()
    (tmp_path / "tmp_build/generated.cpp").write_text("cache")
    assert runtime_hashes(tmp_path) == original
    (tmp_path / "train_pidg.py").write_text("# modified\n")
    assert runtime_hashes(tmp_path) != original


def test_real_a0_audit():
    project = Path(__file__).resolve().parents[1]
    # CUDA/source/data are local optional assets on a fresh checkout.
    if not all((project / name).exists() for name in (
            "third_party/PIDG/train_pidg.py", "data/PIDG/mechanics_cloth/audit.json",
            "runs/pidg_official/rubber_duck/reproduction.json",
            "variants/phase00_baseline/source_manifest.json")):
        pytest.skip("Local PIDG reproduction assets unavailable")
    report = audit(project / "variants/phase00_baseline")
    assert report["source_parity"]
    assert len(report["formal_baseline"]) == 5
