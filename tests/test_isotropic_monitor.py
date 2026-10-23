"""Real log edge cases for a read-only monitor: partial writes and resumptions."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('isotropic_chart_monitor', ROOT / 'scripts/monitor_isotropic.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_partial_jsonl_record_waits_then_updates_same_iteration_on_resume(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    record = {'iteration': 1000, 'metrics': {'PSNR': 28., 'SSIM': .97, 'LPIPS': .03, 'MS-SSIM': .96}}
    text = json.dumps(record)
    path.write_text(text[:30])
    reader = module.History(path, 'metric')
    assert reader.update() == []
    with path.open('a') as stream:
        stream.write(text[30:] + '\n')
    assert reader.update() == [record]
    assert reader.update() == [record]
    record['metrics']['PSNR'] = 29.
    with path.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    assert reader.update() == [record]


def test_carriage_return_progress_loss_and_truncation(tmp_path):
    path = tmp_path / 'train.log'
    path.write_bytes(b'Loading data\rTraining progress:  2%|bar| 1000/50000 [1:00, Loss=3.2e-03, pts=10]\r')
    reader = module.History(path, 'train')
    assert reader.update() == [{'iteration': 1000, 'loss': .0032}]
    assert reader.latest == 1000 and reader.target == 50000
    path.write_bytes(b'Training progress: 10/100 [Loss=0.2]\r')
    assert reader.update() == [{'iteration': 10, 'loss': .2}]
    assert reader.target == 100


def test_monitor_ignores_nonfinite_or_corrupt_complete_records(tmp_path):
    path = tmp_path / 'isotropic.jsonl'
    path.write_text('bad json\n' + json.dumps({'iteration': 10, 'loss': float('nan'), 'weighted_loss': .1}) + '\n' + json.dumps({'iteration': 20, 'loss': .3, 'weighted_loss': .0003}) + '\n')
    assert module.History(path, 'isotropic').update() == [{'iteration': 20, 'loss': .3, 'weighted_loss': .0003}]


def test_chart_downsampling_preserves_spikes_and_endpoints():
    points = [{'iteration': i, 'loss': 100. if i == 501 else .01} for i in range(5000)]
    reduced = module.extrema_sample(points, 1200)
    assert len(reduced) <= 1200
    assert reduced[0] == points[0] and reduced[-1] == points[-1]
    assert points[501] in reduced


def test_finished_controller_is_not_reported_as_training(tmp_path, monkeypatch):
    runs, reports = tmp_path / 'runs', tmp_path / 'reports'
    run = runs / 'I2/rubber_duck'
    run.mkdir(parents=True); reports.mkdir()
    (runs / 'controller.json').write_text(json.dumps({'status': 'completed', 'pid': 123}))
    (runs / 'active.json').write_text(json.dumps({'run': str(run), 'pid': 124, 'phase': 'metrics'}))
    monkeypatch.setattr(module, 'RUNS', runs)
    monkeypatch.setattr(module, 'REPORTS', reports)
    monkeypatch.setattr(module, 'T3', tmp_path / 'no-t3')
    monkeypatch.setattr(module, 'BASELINE', tmp_path / 'no-baseline')
    monkeypatch.setattr(module, 'alive', lambda _: False)
    monkeypatch.setattr(module, 'gpu_status', lambda: [])
    monitor = module.Monitor(); monitor.update()
    assert monitor.snapshot['status'] == 'completed'
    assert monitor.snapshot['metrics'] == []
    assert (reports / 'charts_data.js').read_text().startswith('window.PIDG_MONITOR_DATA=')
