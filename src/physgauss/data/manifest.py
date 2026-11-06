from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from ..rendering import Camera


@dataclass
class Frame:
    index: int
    time: torch.Tensor
    camera: Camera
    rgb: torch.Tensor
    rgb_valid: torch.Tensor
    depth: torch.Tensor | None = None
    depth_valid: torch.Tensor | None = None
    flow: torch.Tensor | None = None
    flow_valid: torch.Tensor | None = None
    next_time: torch.Tensor | None = None
    next_camera: Camera | None = None


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def require_shape(array, shape, name):
    if array.shape != tuple(shape):
        raise ValueError(f"{name}: expected {tuple(shape)}, got {array.shape}")


class SceneDataset:
    """Validated common interchange format; source benchmark splits remain explicit."""
    def __init__(self, manifest_path, device="cpu"):
        self.path = Path(manifest_path).resolve()
        self.metadata = json.loads(self.path.read_text(encoding="utf-8"))
        m = self.metadata
        if m.get("schema_version") != 1:
            raise ValueError("Only manifest schema_version=1 is supported")
        if m.get("protocol") not in {"image_only", "ct_assisted"}:
            raise ValueError("protocol must be image_only or ct_assisted")
        if m.get("dataset_kind") not in {"synthetic", "public", "trachea"}:
            raise ValueError("dataset_kind must be synthetic, public or trachea")
        if m["protocol"] == "image_only" and m.get("geometry_source") not in {"image", "synthetic"}:
            raise ValueError("image_only protocol excludes CT geometry")
        if m["protocol"] == "ct_assisted" and (m.get("geometry_source") != "ct" or m["dataset_kind"] != "trachea"):
            raise ValueError("CT-assisted geometry is reserved for the trachea setting")
        if m.get("time_unit") != "seconds" or m.get("length_unit") not in {"meters", "normalized"}:
            raise ValueError("Specify time_unit=seconds and length_unit=meters or normalized")
        if m["length_unit"] == "normalized" and m["dataset_kind"] != "synthetic":
            if not isinstance(m.get("meters_per_unit"), (int, float)) or m["meters_per_unit"] <= 0:
                raise ValueError("Real normalized data requires a positive meters_per_unit")
        self.frames_path = (self.path.parent / m["frames_file"]).resolve()
        self.cloud_path = (self.path.parent / m["canonical_file"]).resolve()
        arrays = load_npz(self.frames_path)
        for key in ("rgb", "K", "world_to_camera", "times", "train", "test"):
            if key not in arrays:
                raise ValueError(f"Missing required frame array: {key}")
        rgb = arrays["rgb"]
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError("rgb must have shape [T,H,W,3]")
        count, height, width, _ = rgb.shape
        if count < 2 or min(height, width) < 3:
            raise ValueError("At least 2 frames and 3x3 images are required")
        for key, shape in {"K": (count, 3, 3), "world_to_camera": (count, 4, 4),
                           "times": (count,), "train": (count,), "test": (count,)}.items():
            require_shape(arrays[key], shape, key)
        for key in ("rgb", "K", "world_to_camera", "times"):
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"{key} must be finite")
        if rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("RGB must be float in [0,1]")
        if np.any(np.diff(arrays["times"]) <= 0):
            raise ValueError("times must be strictly increasing in seconds")
        for key in ("train", "test"):
            if arrays[key].dtype != np.bool_:
                raise ValueError(f"{key} must be a boolean mask")
        if not arrays["train"].any() or not arrays["test"].any() or np.any(arrays["train"] & arrays["test"]):
            raise ValueError("train/test must be nonempty and disjoint")
        K, extrinsic = arrays["K"], arrays["world_to_camera"]
        if np.any(K[:, 0, 0] <= 0) or np.any(K[:, 1, 1] <= 0):
            raise ValueError("Camera focal lengths must be positive")
        if not np.allclose(K[:, 0, 1], 0) or not np.allclose(K[:, 1, 0], 0) or not np.allclose(K[:, 2], [0, 0, 1]):
            raise ValueError("Reference renderer requires zero-skew pinhole intrinsics")
        R = extrinsic[:, :3, :3]
        if (not np.allclose(extrinsic[:, 3], [0, 0, 0, 1], atol=1e-4)
                or not np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-4)
                or not np.allclose(np.linalg.det(R), 1, atol=1e-4)):
            raise ValueError("world_to_camera must be rigid, right-handed transforms")
        arrays.setdefault("rgb_valid", np.ones((count, height, width), dtype=bool))
        for key, shape in {"rgb_valid": (count, height, width),
                           "depth": (count, height, width), "flow": (count, height, width, 2)}.items():
            if key in arrays:
                require_shape(arrays[key], shape, key)
        if arrays["rgb_valid"].dtype != np.bool_ or np.any(arrays["rgb_valid"].reshape(count, -1).sum(1) == 0):
            raise ValueError("Each frame must have boolean rgb_valid with some valid pixels")
        for key in ("depth", "flow"):
            if key not in arrays:
                continue
            mask_key = f"{key}_valid"
            if mask_key not in arrays:
                raise ValueError(f"{key} requires an explicit {mask_key}")
            mask = arrays[mask_key]
            require_shape(mask, (count, height, width), mask_key)
            if mask.dtype != np.bool_ or not np.isfinite(arrays[key][mask]).all():
                raise ValueError(f"{mask_key} must be boolean and supervised {key} finite")
            if key == "depth" and np.any(arrays[key][mask] <= 0):
                raise ValueError("Valid depth must be positive camera-z depth")
            arrays[key] = np.where(mask[..., None] if key == "flow" else mask, arrays[key], 0)
        if "flow" in arrays:
            if "flow_next" not in arrays:
                raise ValueError("flow requires explicit flow_next frame indices")
            require_shape(arrays["flow_next"], (count,), "flow_next")
            if arrays["flow_next"].dtype.kind not in "iu":
                raise ValueError("flow_next must contain integers")
            for i, nxt in enumerate(arrays["flow_next"]):
                if nxt != -1 and not i < nxt < count:
                    raise ValueError("flow_next must be -1 or a later frame index")
                if nxt == -1 and arrays["flow_valid"][i].any():
                    raise ValueError("No valid flow is allowed without a target frame")
        cloud = load_npz(self.cloud_path)
        if "xyz" not in cloud:
            raise ValueError("canonical_file requires xyz")
        xyz = cloud["xyz"]
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
            raise ValueError("canonical xyz must be finite [N,3], N>0")
        cloud.setdefault("rgb", np.full_like(xyz, 0.5))
        cloud.setdefault("scales", np.full_like(xyz, 0.02))
        for key in ("rgb", "scales"):
            require_shape(cloud[key], xyz.shape, f"canonical {key}")
            if not np.isfinite(cloud[key]).all():
                raise ValueError(f"canonical {key} must be finite")
        if cloud["rgb"].min() < 0 or cloud["rgb"].max() > 1 or np.any(cloud["scales"] <= 0):
            raise ValueError("canonical rgb must be [0,1], scales positive")
        cloud = {k: np.asarray(cloud[k], dtype=np.float32) for k in ("xyz", "rgb", "scales")}
        # Convert normalized real coordinates to SI, including translations and depth.
        scale = float(m.get("meters_per_unit", 1.0))
        if m["length_unit"] == "normalized" and m["dataset_kind"] != "synthetic":
            cloud["xyz"] *= scale
            cloud["scales"] *= scale
            arrays["world_to_camera"] = arrays["world_to_camera"].astype(np.float32)
            arrays["world_to_camera"][:, :3, 3] *= scale
            if "depth" in arrays:
                arrays["depth"] = arrays["depth"].astype(np.float32)*scale
        self.arrays = {key: torch.as_tensor(value, device=device, dtype=(
            torch.bool if value.dtype == np.bool_ else torch.long if key == "flow_next" else torch.float32
        )) for key, value in arrays.items()}
        self.xyz, self.colors, self.scales = [torch.tensor(cloud[k], dtype=torch.float32, device=device)
                                             for k in ("xyz", "rgb", "scales")]
        self.train_indices = np.flatnonzero(arrays["train"]).tolist()
        self.test_indices = np.flatnonzero(arrays["test"]).tolist()
        self.height, self.width, self.count = height, width, count

    def camera(self, index):
        return Camera(self.arrays["K"][index], self.arrays["world_to_camera"][index], self.height, self.width)

    def frame(self, index, training=False):
        a = self.arrays
        f = Frame(index, a["times"][index], self.camera(index), a["rgb"][index], a["rgb_valid"][index])
        if "depth" in a:
            f.depth, f.depth_valid = a["depth"][index], a["depth_valid"][index]
        if "flow" in a:
            nxt = int(a["flow_next"][index])
            # Flow inferred from held-out images cannot supervise a training frame.
            if nxt >= 0 and (not training or nxt in self.train_indices):
                f.flow, f.flow_valid = a["flow"][index], a["flow_valid"][index]
                f.next_time, f.next_camera = a["times"][nxt], self.camera(nxt)
        return f

    def fingerprint(self):
        digest = hashlib.sha256()
        for path in (self.path, self.frames_path, self.cloud_path):
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
