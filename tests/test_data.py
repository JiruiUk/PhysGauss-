import json
import numpy as np
import pytest
import torch
from physgauss.data import SceneDataset
from physgauss.data.manifest import load_npz


def modify_metadata(path, **updates):
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(updates)
    path.write_text(json.dumps(metadata), encoding="utf-8")


def test_public_protocol_rejects_ct_initialization(scene_path):
    modify_metadata(scene_path, geometry_source="ct", dataset_kind="public")
    with pytest.raises(ValueError, match="excludes CT"):
        SceneDataset(scene_path)


def test_ct_assisted_protocol_accepts_canonical_ct_cloud(scene_path):
    modify_metadata(scene_path, protocol="ct_assisted", geometry_source="ct", dataset_kind="trachea",
                    length_unit="meters")
    assert SceneDataset(scene_path).metadata["protocol"] == "ct_assisted"


def test_no_test_image_flow_is_used_for_training(dataset):
    assert dataset.frame(1).flow is not None  # targets held-out frame 2
    assert dataset.frame(1, training=True).flow is None
    assert dataset.frame(0, training=True).flow is not None


def test_rejects_overlapping_splits(scene_path):
    frames_path = scene_path.parent / "frames.npz"
    arrays = load_npz(frames_path)
    arrays["train"][2] = True
    np.savez_compressed(frames_path, **arrays)
    with pytest.raises(ValueError, match="disjoint"):
        SceneDataset(scene_path)


def test_invalid_masked_depth_is_sanitized(scene_path):
    frames_path = scene_path.parent / "frames.npz"
    arrays = load_npz(frames_path)
    arrays["depth"][~arrays["depth_valid"]] = np.nan
    np.savez_compressed(frames_path, **arrays)
    assert torch.isfinite(SceneDataset(scene_path).arrays["depth"]).all()


def test_normalized_real_coordinates_are_converted_to_meters(scene_path):
    before = SceneDataset(scene_path)
    modify_metadata(scene_path, dataset_kind="trachea", protocol="ct_assisted", geometry_source="ct",
                    meters_per_unit=0.01)
    after = SceneDataset(scene_path)
    torch.testing.assert_close(after.xyz, before.xyz*0.01)
    torch.testing.assert_close(after.scales, before.scales*0.01)
    torch.testing.assert_close(after.arrays["depth"], before.arrays["depth"]*0.01)


def test_manifest_fingerprint_detects_modified_data(scene_path):
    before = SceneDataset(scene_path).fingerprint()
    modify_metadata(scene_path, name="renamed")
    assert SceneDataset(scene_path).fingerprint() != before
