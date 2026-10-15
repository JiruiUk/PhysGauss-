"""Audit isolated A0 source and completed historical PIDG benchmarks."""
import argparse
from pathlib import Path
from pidg_benchmark import (PROJECT, SCENES, completed_metrics, dataset_hashes,
                            read_json, runtime_hashes, sha256, write_json)


def audit(variant):
    variant = Path(variant)
    manifest = read_json(variant / "source_manifest.json")
    original = runtime_hashes(PROJECT / "third_party/PIDG")
    copied = runtime_hashes(variant / "code")
    if original != manifest["files"] or copied != original:
        raise ValueError("A0 runtime source differs from the frozen patched baseline")
    for name, expected in manifest["shared_helpers"].items():
        if sha256(PROJECT / name) != expected:
            raise ValueError(f"Shared runtime dependency changed: {name}")
    formal = []
    for scene in SCENES:
        run = PROJECT / "runs/pidg_official" / scene
        metadata, metrics = completed_metrics(run)
        if (metadata["iterations"], metadata["resolution_width"], metadata["seed"],
                metadata["smoke_only"], metadata["profile"]) != (50000, 1600, 0, False, "official"):
            raise ValueError(f"Historical benchmark protocol differs: {scene}")
        audit_data = read_json(PROJECT / "data/PIDG" / scene / "audit.json")
        if audit_data["missing"] or audit_data["shape_mismatches"]:
            raise ValueError(f"Failed dataset audit: {scene}")
        formal.append({"scene": scene, "run": str(run), "metrics": metrics,
                       "dataset_hashes": dataset_hashes(scene),
                       "metadata_sha256": sha256(run / "reproduction.json"),
                       "results_sha256": sha256(run / "results.json")})
    return {"phase": 0, "source_parity": True, "runtime_file_count": len(original),
            "formal_baseline": formal,
            "note": "Historical runs lack runtime hashes; hashes freeze current patched code, not retrospective provenance."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=Path, default=PROJECT / "variants/phase00_baseline")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.variant)
    if args.output:
        write_json(args.output, report)
    print(f"PASS: {report['runtime_file_count']} identical runtime files; 5 completed official benchmarks")


if __name__ == "__main__":
    main()
