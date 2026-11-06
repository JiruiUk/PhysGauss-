"""Fetch the official VGG-16 weights required by the released LPIPS evaluator."""
import hashlib
from pathlib import Path
import urllib.request

project = Path(__file__).resolve().parents[2]
destination = project / "reproduction/torch-cache/hub/checkpoints/vgg16-397923af.pth"
destination.parent.mkdir(parents=True, exist_ok=True)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


if destination.is_file() and digest(destination).startswith("397923af"):
    print("VGG-16 checkpoint already verified.", flush=True)
else:
    temporary = destination.with_suffix(".part")
    with urllib.request.urlopen("https://download.pytorch.org/models/vgg16-397923af.pth", timeout=180) as source, temporary.open("wb") as output:
        for chunk in iter(lambda: source.read(4*1024*1024), b""):
            output.write(chunk)
    sha = digest(temporary)
    if not sha.startswith("397923af"):
        raise ValueError(f"VGG-16 hash check failed: {sha}")
    temporary.replace(destination)
    print(f"Official VGG-16 verified: {sha}", flush=True)
