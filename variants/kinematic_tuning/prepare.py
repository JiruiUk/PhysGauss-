"""Independent source fork; all previous experiment sources stay frozen."""
from pathlib import Path
import shutil
from variants.metric_tuning.prepare import prepare as prepare_metric, replace_once
from variants.metric_tuning.run import runtime_hashes, read_json, write_json

TARGET = Path(__file__).parent/'code'


def prepare():
    source = prepare_metric()
    original = runtime_hashes(source)
    receipt = TARGET.parent/'source_manifest.json'
    if TARGET.exists():
        saved = read_json(receipt)
        if saved['parent'] != original or saved['fork'] != runtime_hashes(TARGET):
            raise ValueError('Kinematic fork or parent runtime changed')
        return TARGET
    for relative in original:
        destination = TARGET/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, destination)
    path = TARGET/'train_pidg.py'
    text = path.read_text(encoding='utf-8')
    text = replace_once(text, 'from variants.metric_tuning.runtime import Tuning',
                        'from variants.kinematic_tuning.runtime import Tuning')
    anchor = '    tuning.after_scene(gaussians, deform)\n'
    text = replace_once(text, anchor, anchor+'    tuning.configure_scene(scene, dataset.source_path)\n')
    anchor = '            loss = tuning.image_loss(iteration, image, gt_image, opt) + reg\n'
    text = replace_once(text, anchor, anchor+
        '            kinematic_loss = tuning.kinematic_loss(iteration, gaussians, deform, fid)\n'
        '            if kinematic_loss is not None:\n'
        '                loss = loss + kinematic_loss\n')
    text = replace_once(text, '            t_scalar = fid.float() / total_frame',
                        '            t_scalar = tuning.physics_time(fid, total_frame)')
    text = replace_once(text, '                dt=1/total_frame,',
                        '                dt=tuning.time_step(total_frame),')
    path.write_text(text, encoding='utf-8')
    path = TARGET/'render.py'
    text = path.read_text(encoding='utf-8')
    anchor = '        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)\n'
    text = replace_once(text, anchor, anchor+
        '        from variants.kinematic_tuning.timeline import configure_scene_from_env\n'
        '        configure_scene_from_env(scene, dataset.source_path)\n')
    path.write_text(text, encoding='utf-8')
    assert runtime_hashes(source) == original
    write_json(receipt, {'parent': original, 'fork': runtime_hashes(TARGET)})
    return TARGET


if __name__ == '__main__': print(prepare())
