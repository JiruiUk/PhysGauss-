"""Independent trainer+renderer fork; parent and flags-off renderer remain intact."""
from pathlib import Path
import shutil
from variants.coverage_tuning.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes,read_json,write_json

HERE=Path(__file__).parent
TARGET=HERE/'code'


def prepare():
    source=parent_prepare();parent=runtime_hashes(source);receipt=HERE/'source_manifest.json'
    if TARGET.exists():
        if read_json(receipt)!={'parent':parent,'fork':runtime_hashes(TARGET)}:raise ValueError('Color-repair fork changed')
        return TARGET
    for relative in parent:
        path=TARGET/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,path)
    path=TARGET/'train_pidg.py'
    path.write_text(replace_once(path.read_text(encoding='utf-8'),
        'from variants.coverage_tuning.runtime import Tuning','from variants.color_repair.runtime import Tuning'),encoding='utf-8')
    path=TARGET/'gaussian_renderer/__init__.py';text=path.read_text(encoding='utf-8')
    text=replace_once(text,'    if override_color is None:\n',
        '    from variants.color_repair.color import enabled as positive_enabled, colors as positive_colors\n'
        '    if override_color is None and positive_enabled():\n'
        '        colors_precomp = positive_colors(pc, means3D, viewpoint_camera)\n'
        '    elif override_color is None:\n')
    path.write_text(text,encoding='utf-8')
    assert runtime_hashes(source)==parent
    write_json(receipt,{'parent':parent,'fork':runtime_hashes(TARGET)})
    return TARGET
