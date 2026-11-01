"""Isolated H3 trainer with one opt-in loss import; no baseline edits."""
from pathlib import Path
import shutil
from variants.integrity_tuning.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes,read_json,write_json
HERE=Path(__file__).parent


def prepare():
    source=parent_prepare();parent=runtime_hashes(source);target=HERE/'code';receipt=HERE/'source_manifest.json'
    if target.exists():
        if read_json(receipt)!={'parent':parent,'fork':runtime_hashes(target)}:raise ValueError('Temporal repair fork changed')
        return target
    for relative in parent:
        path=target/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,path)
    file=target/'train_pidg.py'
    file.write_text(replace_once(file.read_text(encoding='utf-8'),'from variants.integrity_tuning.runtime import Tuning',
        'from variants.temporal_repair.runtime import Tuning'),encoding='utf-8')
    file.write_text(replace_once(file.read_text(encoding='utf-8'),'        if dataset.load2gpu_on_the_fly:\n            viewpoint_cam.load2device()\n        fid = viewpoint_cam.fid',
        '        viewpoint_cam = tuning.select_training_camera(iteration, viewpoint_cam, scene.getTrainCameras())\n        if dataset.load2gpu_on_the_fly:\n            viewpoint_cam.load2device()\n        fid = viewpoint_cam.fid'),encoding='utf-8')
    renderer=target/'gaussian_renderer/__init__.py'
    renderer.write_text(replace_once(renderer.read_text(encoding='utf-8'),'        rotations = pc.get_rotation + d_rotation',
        '        from variants.temporal_repair.covariance import rotation_for_render\n        rotations = rotation_for_render(pc.get_rotation + d_rotation)'),encoding='utf-8')
    write_json(receipt,{'parent':parent,'fork':runtime_hashes(target)})
    return target
