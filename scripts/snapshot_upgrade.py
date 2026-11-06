"""Freeze a runnable stage; never replace an existing variant or run."""
import argparse
from pathlib import Path
import shutil
from pidg_benchmark import PROJECT, runtime_hashes, write_json

NAMES = {1: "continuous_time", 2: "kinematic_closure", 3: "acceleration",
         4: "deformation_gradient", 5: "conservation", 6: "constitutive",
         7: "material_field", 8: "closed_momentum", 9: "weak_form", 10: "material_moe",
         11: "neural_residual", 12: "contact", 13: "uncertainty_flow",
         14: "continuous_se3", 15: "symplectic_rollout"}


def snapshot(phase, config):
    target = PROJECT / "variants" / f"phase{phase:02d}_{NAMES[phase]}"
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(PROJECT / "variants/development/code", target / "code",
                    ignore=shutil.ignore_patterns("__pycache__", "tmp_build", "*.pyc"))
    write_json(target / "config.json", config)
    write_json(target / "source_manifest.json", {"phase": phase, "files": runtime_hashes(target / "code")})
    write_json(target / "stage.json", {"phase": phase, "name": NAMES[phase], "status": "implemented",
                "unit_tests": "pending", "smoke": "pending", "long_training": "pending",
                "benchmark": {"scenes": ["mechanics_cloth", "balls_reaction", "dry_ice", "motion_kuro", "rubber_duck"],
                              "iterations": 50000, "width": 1600, "seed": 0}})
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=int, choices=range(1,16))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from pidg_benchmark import read_json
    print(snapshot(args.phase, read_json(args.config)))
