"""An isolated copy of the completed metric-tuning runtime; old sources immutable."""
from pathlib import Path
import shutil
from variants.metric_tuning.prepare import prepare as prepare_metric
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes, read_json, write_json

TARGET = Path(__file__).parent / 'code'


def prepare():
    source = prepare_metric()
    original = runtime_hashes(source)
    receipt = TARGET.parent / 'source_manifest.json'
    if TARGET.exists():
        saved = read_json(receipt)
        if saved['parent'] != original or saved['fork'] != runtime_hashes(TARGET):
            raise ValueError('Isotropic fork or parent runtime changed')
        return TARGET
    for relative in original:
        destination = TARGET / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    path = TARGET / 'train_pidg.py'
    text = path.read_text(encoding='utf-8')
    text = replace_once(text, 'from variants.metric_tuning.runtime import Tuning',
                        'from variants.isotropic_tuning.runtime import Tuning')
    anchor = '            loss = tuning.image_loss(iteration, image, gt_image, opt) + reg\n'
    text = replace_once(text, anchor, anchor +
                        '            iso_loss = tuning.regularization(iteration, gaussians, d_scaling, pipe)\n'
                        '            if iso_loss is not None:\n'
                        '                loss = loss + iso_loss\n')
    path.write_text(text, encoding='utf-8')
    assert runtime_hashes(source) == original
    write_json(receipt, {'parent': original, 'fork': runtime_hashes(TARGET)})
    return TARGET


if __name__ == '__main__':
    print(prepare())
