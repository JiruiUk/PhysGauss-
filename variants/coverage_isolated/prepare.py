"""Frozen coverage fork with only its Tuning import changed."""
from pathlib import Path
import shutil
from variants.coverage_tuning.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes,read_json,write_json

HERE = Path(__file__).parent
TARGET = HERE/'code'


def prepare():
    source = parent_prepare(); original = runtime_hashes(source)
    receipt = HERE/'source_manifest.json'
    if TARGET.exists():
        if read_json(receipt)!={'parent':original,'fork':runtime_hashes(TARGET)}:
            raise ValueError('Isolated-component source changed')
        return TARGET
    for relative in original:
        path = TARGET/relative; path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/relative,path)
    path = TARGET/'train_pidg.py'
    text = replace_once(path.read_text(encoding='utf-8'),
        'from variants.coverage_tuning.runtime import Tuning','from variants.coverage_isolated.runtime import Tuning')
    path.write_text(text,encoding='utf-8')
    assert runtime_hashes(source)==original
    write_json(receipt,{'parent':original,'fork':runtime_hashes(TARGET)})
    return TARGET
