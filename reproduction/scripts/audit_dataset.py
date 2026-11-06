import argparse
import json
from pathlib import Path
import struct
from PIL import Image
import numpy as np


def audit(scene):
    scene = Path(scene).resolve()
    train = json.loads((scene / "transforms_train.json").read_text())
    test = json.loads((scene / "transforms_test.json").read_text())
    train_ids = [f["file_path"] for f in train["frames"]]
    test_ids = [f["file_path"] for f in test["frames"]]
    if set(train_ids) & set(test_ids):
        raise ValueError("Official train/test image IDs overlap")
    missing, mismatches, gaps = [], [], []
    for i, frame in enumerate(train["frames"]):
        image_path = scene / (frame["file_path"] + ".png")
        if not image_path.is_file():
            missing.append(str(image_path))
            continue
        stem = image_path.stem
        with Image.open(image_path) as image:
            width, height = image.size
        depth_path = scene / "depth_distill" / (stem + ".npy")
        if not depth_path.is_file():
            missing.append(str(depth_path))
        else:
            depth = np.load(depth_path, mmap_mode="r")
            if depth.shape != (height, width):
                mismatches.append({"frame": stem, "depth": list(depth.shape), "image": [height, width]})
            if not np.isfinite(depth).all():
                mismatches.append({"frame": stem, "depth": "nonfinite"})
        for direction in ("fwd", "bwd"):
            flow = scene / "flows_flo" / f"flow_{direction}_{stem}.flo"
            boundary = (direction == "fwd" and int(stem) == max(int(Path(x).stem) for x in train_ids+test_ids)) or (direction == "bwd" and int(stem) == min(int(Path(x).stem) for x in train_ids+test_ids))
            if not flow.is_file() and not boundary:
                missing.append(str(flow))
            if flow.is_file():
                with flow.open("rb") as source:
                    magic, w, h = struct.unpack("<fii", source.read(12))
                if magic != 202021.25 or (w, h) != (width, height):
                    mismatches.append({"frame": stem, "flow": [h, w], "image": [height, width]})
        if i+1 < len(train_ids) and int(Path(train_ids[i+1]).stem) != int(stem)+1:
            gaps.append([frame["file_path"], train_ids[i+1]])
    for frame in test["frames"]:
        if not (scene / (frame["file_path"] + ".png")).is_file():
            missing.append(frame["file_path"])
    mask_directory = next((d for d in ("motion_mask", "mask") if (scene/d).is_dir()), None)
    report = {"scene": scene.name, "train_count": len(train_ids), "test_count": len(test_ids),
              "mask_directory": mask_directory, "missing": missing, "shape_mismatches": mismatches,
              "nonconsecutive_train_pairs": gaps,
              "upstream_note": "Source next_cam links the next TRAIN frame, while supplied flow is adjacent RAW-frame flow. These gaps are recorded, not silently relabelled."}
    (scene / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if missing or mismatches:
        raise ValueError(f"Data audit failed: {len(missing)} missing, {len(mismatches)} mismatched")
    print(json.dumps({k: v for k, v in report.items() if k != "nonconsecutive_train_pairs"}, indent=2))
    print(f"Recorded nonconsecutive train pairs: {len(gaps)}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    audit(parser.parse_args().scene)
