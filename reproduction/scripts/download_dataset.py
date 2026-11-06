"""Download the author's immutable PIDG release, validating file hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import urllib.parse
import urllib.request

REPO = "HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting"
REVISION = "46b23520acaaf5d58938fb19b50a3a62d12c55df"
SCENES = ["mechanics_cloth", "balls_reaction", "dry_ice", "motion_kuro", "rubber_duck"]


def required(name):
    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or ".." in parts or len(parts) < 2:
        return False
    if len(parts) == 2:
        return path.name in {"transforms_train.json", "transforms_test.json", "transforms_all.json", "points3d.ply"}
    return ((parts[1] == "train" and path.suffix == ".png")
            or (parts[1] == "depth_distill" and path.suffix == ".npy")
            or (parts[1] == "flows_flo" and path.suffix == ".flo")
            or (parts[1] in {"mask", "motion_mask"} and path.suffix == ".png"))


def matches(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    sha = hashlib.sha256() if spec.get("lfs") else hashlib.sha1()
    if not spec.get("lfs"):
        sha.update(f"blob {spec['size']}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4*1024*1024), b""):
            sha.update(chunk)
    expected = spec["lfs"]["sha256"] if spec.get("lfs") else spec["blobId"]
    return sha.hexdigest() == expected


def download(root, spec):
    name = spec["rfilename"]
    destination = root.joinpath(*PurePosixPath(name).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if matches(destination, spec):
        return spec["size"], "cached"
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(5):
        try:
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{urllib.parse.quote(name)}?download=true"
            request = urllib.request.Request(url, headers={"User-Agent": "PhysGauss-PIDG-reproduction/0.1"})
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                for chunk in iter(lambda: response.read(4*1024*1024), b""):
                    output.write(chunk)
            if not matches(temporary, spec):
                raise ValueError(f"Size/hash mismatch: {name}")
            temporary.replace(destination)
            return spec["size"], "downloaded"
        except Exception as error:
            if attempt == 4:
                raise RuntimeError(f"Download failed: {name}: {error}") from error
            time.sleep(min(30, 2**attempt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest["sha"] != REVISION:
        raise ValueError("Manifest revision is not the recorded official release")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    all_records = []
    for scene in args.scenes:
        files = [f for f in manifest["siblings"] if f["rfilename"].startswith(scene+"/") and required(f["rfilename"])]
        total = sum(f["size"] for f in files)
        print(f"{scene}: {len(files)} required files, {total/1e9:.3f} GB", flush=True)
        completed = 0
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download, root, spec): spec for spec in files}
            for future in as_completed(futures):
                size, status = future.result()
                spec = futures[future]
                completed += 1
                all_records.append({"path": spec["rfilename"], "bytes": size, "status": status,
                                    "hash": spec.get("lfs", {}).get("sha256", spec["blobId"])})
                if completed % 25 == 0 or completed == len(files):
                    print(f"{scene}: verified {completed}/{len(files)}, elapsed={time.monotonic()-started:.0f}s", flush=True)
        report = {"repo": REPO, "revision": REVISION, "scene": scene,
                  "files": [r for r in all_records if r["path"].startswith(scene+"/")]}
        (root / scene / "download_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("All requested scene files downloaded and hash verified.", flush=True)


if __name__ == "__main__":
    main()
