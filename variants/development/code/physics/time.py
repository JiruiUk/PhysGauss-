"""Frame indices normalize once; published continuous observation times stay floats."""
import math
import torch


def normalize_time(frame_idx, num_frames):
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    return frame_idx / max(num_frames - 1, 1)


def observation_time(value):
    if torch.is_tensor(value):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("Observation time must be a finite floating tensor")
        return value
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite observation time")
    return value


def material_time(fid, train_count, mode="continuous"):
    fid = observation_time(fid)
    if mode == "continuous":
        return fid
    if mode == "legacy":
        return fid / train_count
    raise ValueError(f"Unknown time mode: {mode}")
