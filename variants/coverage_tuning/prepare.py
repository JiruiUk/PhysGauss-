"""Independent runtime copy: three optional hooks; historical forks stay frozen."""
from pathlib import Path
import shutil
from variants.v3_iso_scratch.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes, read_json, write_json

HERE = Path(__file__).parent
TARGET = HERE/'code'


def prepare():
    source = parent_prepare(); original = runtime_hashes(source)
    receipt = HERE/'source_manifest.json'
    if TARGET.exists():
        if read_json(receipt) != {'parent':original,'fork':runtime_hashes(TARGET)}:
            raise ValueError('Coverage fork source changed')
        return TARGET
    for relative in original:
        path = TARGET/relative; path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/relative,path)
    path = TARGET/'train_pidg.py'; text = path.read_text(encoding='utf-8')
    text = replace_once(text,'from variants.v3_iso_scratch.runtime import Tuning',
                        'from variants.coverage_tuning.runtime import Tuning')
    text = replace_once(text,'        loss.backward()\n',
        '        coverage_loss = tuning.coverage_loss(iteration, viewpoint_cam, render_pkg_re, gaussians, d_xyz)\n'
        '        if coverage_loss is not None:\n            loss = loss + coverage_loss\n\n        loss.backward()\n')
    text = replace_once(text,'            tuning.request_stop(iteration)\n',
                        '            tuning.after_update(iteration, gaussians)\n            tuning.request_stop(iteration)\n')
    path.write_text(text,encoding='utf-8')
    # Alpha export is no-grad diagnostics, separate from the original RGB metrics.
    path = TARGET/'render.py'; text = path.read_text(encoding='utf-8')
    text = replace_once(text,'        total_time += time.time() - t\n        rendering = results["render"]\n',
        '        total_time += time.time() - t\n        rendering = results["render"]\n'
        '        if os.environ.get("PIDG_EXPORT_ALPHA") == "1":\n'
        '            alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "alpha")\n'
        '            makedirs(alpha_path, exist_ok=True)\n'
        '            torchvision.utils.save_image(results["alpha"], os.path.join(alpha_path, f"{idx:05d}.png"))\n')
    path.write_text(text,encoding='utf-8')
    assert runtime_hashes(source) == original
    write_json(receipt,{'parent':original,'fork':runtime_hashes(TARGET)})
    return TARGET
