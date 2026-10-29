"""Independent C1 fork; only the optional Tuning class import changes."""
from pathlib import Path
import shutil
from variants.color_repair.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes,read_json,write_json

HERE=Path(__file__).parent
TARGET=HERE/'code'


def prepare():
    source=parent_prepare();parent=runtime_hashes(source);receipt=HERE/'source_manifest.json'
    if TARGET.exists():
        if read_json(receipt)!={'parent':parent,'fork':runtime_hashes(TARGET)}:raise ValueError('Material fork changed')
        return TARGET
    for relative in parent:
        path=TARGET/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,path)
    path=TARGET/'train_pidg.py'
    path.write_text(replace_once(path.read_text(encoding='utf-8'),
        'from variants.color_repair.runtime import Tuning','from variants.c1_material.runtime import Tuning'),encoding='utf-8')
    assert runtime_hashes(source)==parent
    write_json(receipt,{'parent':parent,'fork':runtime_hashes(TARGET)})
    return TARGET
