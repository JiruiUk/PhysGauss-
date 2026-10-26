"""New source fork: held-out validation only, old experiment runtime untouched."""
from pathlib import Path
import shutil
from variants.kinematic_tuning.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes, read_json, write_json

HERE = Path(__file__).parent
TARGET = HERE/'code'


def prepare():
    source = parent_prepare()
    original = runtime_hashes(source)
    receipt = HERE/'source_manifest.json'
    if TARGET.exists():
        if read_json(receipt) != {'parent': original, 'fork': runtime_hashes(TARGET)}:
            raise ValueError('Resplit source changed')
        return TARGET
    for relative in original:
        path = TARGET/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, path)
    path = TARGET/'train_pidg.py'
    text = path.read_text(encoding='utf-8')
    text = replace_once(text, 'from variants.kinematic_tuning.runtime import Tuning',
                        'from variants.kinematic_resplit.runtime import Tuning')
    text = replace_once(text, 'from variants.metric_tuning.checkpoint import restore_training_state, save_training_state',
                        'from variants.kinematic_resplit.checkpoint import restore_training_state, save_training_state')
    path.write_text(text, encoding='utf-8')
    path = TARGET/'render.py'
    text = replace_once(path.read_text(encoding='utf-8'),
        'from variants.kinematic_tuning.timeline import configure_scene_from_env',
        'from variants.kinematic_resplit.runtime import configure_scene_from_env')
    path.write_text(text, encoding='utf-8')
    assert runtime_hashes(source) == original
    write_json(receipt, {'parent': original, 'fork': runtime_hashes(TARGET)})
    return TARGET
