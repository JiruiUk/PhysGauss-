"""Regression tests for image counts being mistaken for training iterations."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import monitor_kinematic as monitor


def make_run(tmp_path):
    runs, reports = tmp_path/'runs', tmp_path/'reports'
    run = runs/monitor.NAMES[0]/'rubber_duck'
    run.mkdir(parents=True)
    (run/'reproduction.json').write_text(json.dumps({'target_additional_iterations': 50000}))
    return runs, reports, run


def test_training_eval_50_images_never_becomes_iter_50(tmp_path):
    runs, reports, run = make_run(tmp_path)
    (run/'train.log').write_text('Training progress: 24000/50000 [0:10, Loss=0.004]\r'
                               'EVAL_PROGRESS iteration=24000 views=50/50\n')
    arm = monitor.Monitor(runs, reports).arm(monitor.NAMES[0], {}, {})
    assert arm['iteration'] == 24000 and arm['target'] == 50000
    stage = monitor.stage_progress('long', (run/'train.log').read_text())
    assert stage['label'] == '训练中评估' and stage['images'] == 50
    assert stage['iteration'] == 24000


def test_final_checkpoint_preserves_50000_during_rendering(tmp_path, monkeypatch):
    runs, reports, run = make_run(tmp_path)
    (run/'train.log').write_text('Training progress: 49990/50000 [0:10, Loss=0.004]\r')
    (run/'checkpoints').mkdir()
    (run/'checkpoints/resume.json').write_text(json.dumps({'iteration': 50000}))
    (runs/'controller.json').write_text(json.dumps({'status': 'training', 'pid': 123}))
    (runs/'active.json').write_text(json.dumps({'run': str(run), 'id': monitor.NAMES[0], 'pid': 124, 'phase': 'render_best'}))
    (run/'render_best.log').write_text('Rendering progress: 12/50 [0:01]\r')
    monkeypatch.setattr(monitor, 'alive', lambda _: True)
    reader = monitor.Monitor(runs, reports);reader.update()
    assert reader.snapshot['current']['iteration'] == 50000
    assert reader.snapshot['stage']['images'] == 12
    assert reader.snapshot['stage']['total_images'] == 50


def test_completed_eval_returns_to_training_stage():
    text = 'Training progress: 3000/50000 [Loss=0.1]\rEVAL_PROGRESS iteration=3000 views=50/50\nEVAL_METRICS {}\n'
    assert monitor.stage_progress('long', text) == {'label': '训练'}
    assert monitor.stage_progress('long', text+'Training progress: 3010/50000 [Loss=0.1]\r') == {'label': '训练'}


def test_evaluation_iteration_fallback_ignores_unrelated_fractions(tmp_path):
    runs, reports, run = make_run(tmp_path)
    (run/'train.log').write_text('Loading 200/200 [0:01]\nMetric evaluation progress: 50/50 [0:01]\n')
    (run/'metric_history.jsonl').write_text(json.dumps({'iteration': 17000, 'metrics': {
        'PSNR': 28., 'SSIM': .97, 'LPIPS': .03, 'MS-SSIM': .97}})+'\n')
    arm = monitor.Monitor(runs, reports).arm(monitor.NAMES[0], {}, {})
    assert arm['iteration'] == 17000 and arm['iteration_source'] == 'evaluation_iteration'


def test_completed_controller_does_not_show_old_render_stage(tmp_path, monkeypatch):
    runs, reports, run = make_run(tmp_path)
    (runs/'controller.json').write_text(json.dumps({'status': 'completed', 'completed': list(monitor.NAMES)}))
    (runs/'active.json').write_text(json.dumps({'run': str(run), 'id': monitor.NAMES[0], 'phase': 'metrics'}))
    monkeypatch.setattr(monitor, 'alive', lambda _: False)
    reader = monitor.Monitor(runs, reports);reader.update()
    assert reader.snapshot['stage']['label'] == '全部实验完成'
    assert reader.snapshot['status'] == 'completed'
