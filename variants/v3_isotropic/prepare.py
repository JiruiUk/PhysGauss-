"""Frozen V3 resplit fork with an independently disabled-by-default iso hook."""
from pathlib import Path
import shutil
from variants.kinematic_resplit.prepare import prepare as prepare_parent
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes, read_json, write_json

HERE = Path(__file__).parent
TARGET = HERE/'code'


def prepare():
    source = prepare_parent()
    original = runtime_hashes(source)
    receipt = HERE/'source_manifest.json'
    if TARGET.exists():
        if read_json(receipt) != {'parent': original, 'fork': runtime_hashes(TARGET)}:
            raise ValueError('V3+iso source changed')
        return TARGET
    for relative in original:
        destination = TARGET/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, destination)
    path = TARGET/'train_pidg.py'
    text = replace_once(path.read_text(encoding='utf-8'),
        'from variants.kinematic_resplit.runtime import Tuning', 'from variants.v3_isotropic.runtime import Tuning')
    anchor = '            loss = tuning.image_loss(iteration, image, gt_image, opt) + reg\n'
    text = replace_once(text, anchor, anchor+
        '            iso_loss = tuning.regularization(iteration, gaussians, d_scaling, pipe)\n'
        '            if iso_loss is not None:\n'
        '                loss = loss + iso_loss\n')
    path.write_text(text, encoding='utf-8')
    assert runtime_hashes(source) == original
    write_json(receipt, {'parent': original, 'fork': runtime_hashes(TARGET)})
    return TARGET
