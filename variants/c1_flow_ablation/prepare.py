"""Independent C1 fork: optional runtime import and additive observed-flow hook only."""
from pathlib import Path
import shutil
from variants.color_repair.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes, read_json, write_json

HERE = Path(__file__).parent
TARGET = HERE/'code'


def prepare():
    source = parent_prepare(); parent = runtime_hashes(source); receipt = HERE/'source_manifest.json'
    if TARGET.exists():
        if read_json(receipt) != {'parent': parent, 'fork': runtime_hashes(TARGET)}:
            raise ValueError('C1 flow fork changed')
        return TARGET
    for relative in parent:
        path = TARGET/relative; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, path)
    path = TARGET/'train_pidg.py'; text = path.read_text(encoding='utf-8')
    text = replace_once(text, 'from variants.color_repair.runtime import Tuning',
        'from variants.c1_flow_ablation.runtime import Tuning')
    text = replace_once(text, '        coverage_loss = tuning.coverage_loss(',
        '        observed_loss = tuning.observed_loss(iteration, gaussians, deform, viewpoint_cam,\n'
        '            deform_pkgs if iteration >= opt.warm_up else None, render, pipe, background)\n'
        '        if observed_loss is not None:\n'
        '            loss = loss + observed_loss\n\n'
        '        coverage_loss = tuning.coverage_loss(')
    path.write_text(text, encoding='utf-8')
    assert runtime_hashes(source) == parent
    write_json(receipt, {'parent': parent, 'fork': runtime_hashes(TARGET)})
    return TARGET
