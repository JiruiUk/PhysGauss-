"""Frozen C1 copy, with a training-only opt-in Tuning import."""
from pathlib import Path
import shutil
from variants.color_repair.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes,read_json,write_json
HERE=Path(__file__).parent


def prepare():
    source=parent_prepare();parent=runtime_hashes(source);target=HERE/'code';receipt=HERE/'source_manifest.json'
    if target.exists():
        if read_json(receipt)!={'parent':parent,'fork':runtime_hashes(target)}:raise ValueError('Integrity fork changed')
        return target
    for relative in parent:
        path=target/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,path)
    path=target/'train_pidg.py'
    path.write_text(replace_once(path.read_text(encoding='utf-8'),'from variants.color_repair.runtime import Tuning',
        'from variants.integrity_tuning.runtime import Tuning'),encoding='utf-8')
    write_json(receipt,{'parent':parent,'fork':runtime_hashes(target)})
    return target
