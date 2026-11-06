import json
import torch
import pytest
from physgauss.config import TrainConfig
from physgauss.engine.trainer import train
from physgauss.engine.evaluate import evaluate, export_state


def config(steps):
    return TrainConfig(steps=steps, hidden_dim=8, frequencies=1, physics_warmup=0,
                       physics_ramp=1, collocation_count=3, log_every=20)


def test_training_eval_export_and_deterministic_resume(dataset, tmp_path):
    full = train(dataset, config(4), tmp_path / "full")
    partial = train(dataset, config(2), tmp_path / "resumed")
    resumed = train(dataset, config(4), tmp_path / "resumed", resume=partial)
    expected = torch.load(full, weights_only=True)
    actual = torch.load(resumed, weights_only=True)
    for key in expected["model"]:
        torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
    report = evaluate(dataset, resumed, tmp_path / "eval")
    assert report["split"] == "test" and report["benchmark_compatible"] is False
    assert [r["frame"] for r in report["frames"]] == dataset.test_indices
    assert (tmp_path / "eval" / "metrics.json").exists()
    assert report["summary"]["psnr"] > 0
    export = export_state(dataset, resumed, 0.5, tmp_path / "state.npz")
    assert export.exists()
    records = [json.loads(line) for line in (tmp_path / "resumed" / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [1, 2, 3, 4]


def test_resume_rejects_changed_physics_config(dataset, tmp_path):
    checkpoint = train(dataset, config(1), tmp_path / "run")
    changed = config(3)
    changed.density = 2
    with pytest.raises(ValueError, match="same model"):
        train(dataset, changed, tmp_path / "run", resume=checkpoint)


def test_freezing_ct_geometry_keeps_initial_positions(dataset, tmp_path):
    cfg = config(2)
    cfg.freeze_canonical_geometry = True
    checkpoint = train(dataset, cfg, tmp_path / "frozen")
    state = torch.load(checkpoint, weights_only=True)["model"]
    torch.testing.assert_close(state["canonical.means"], dataset.xyz, rtol=0, atol=0)
