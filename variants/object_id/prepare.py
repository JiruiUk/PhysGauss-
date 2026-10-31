"""Copy the frozen C1 runtime and insert opt-in object-motion/checkpoint hooks."""
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
        if read_json(receipt)!={'parent':parent,'fork':runtime_hashes(TARGET)}:raise ValueError('Object fork changed')
        return TARGET
    for relative in parent:
        destination=TARGET/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/relative,destination)
    p=TARGET/'train_pidg.py';s=p.read_text(encoding='utf-8')
    s=replace_once(s,'from variants.color_repair.runtime import Tuning','from variants.object_id.runtime import Tuning')
    s=replace_once(s,'        # Pick a random Camera\n','        tuning.before_iteration(iteration)\n\n        # Pick a random Camera\n')
    p.write_text(s,encoding='utf-8')
    p=TARGET/'scene_PIDG/deform_model.py';s=p.read_text(encoding='utf-8')
    anchor='        self.scale_xyz = torch.tensor(scale_xyz, device="cuda", dtype=torch.float32)\n'
    s=replace_once(s,anchor,anchor+'        from variants.object_id.runtime import configure_deformation\n        configure_deformation(self)\n')
    s=replace_once(s,'    def step(self, xyz, t, fixed_attention=False):\n',
        '    def step(self, xyz, t, fixed_attention=False):\n        object_xyz, object_time = xyz, t\n')
    s=replace_once(s,'        return {\n            "d_xyz": d_xyz,',
        '        from variants.object_id.runtime import apply_warp\n        d_xyz = apply_warp(self, object_xyz, object_time, d_xyz)\n        return {\n            "d_xyz": d_xyz,')
    anchor="        torch.save((self.grid4d.state_dict(), self.deform.state_dict()), os.path.join(out_weights_path, 'deform.pth'))\n"
    s=replace_once(s,anchor,anchor+'        from variants.object_id.runtime import snapshot_save\n        snapshot_save(self, out_weights_path)\n')
    anchor='        self.grid4d.load_state_dict(grid_weight)\n'
    s=replace_once(s,anchor,anchor+'        from variants.object_id.runtime import snapshot_load\n        snapshot_load(self, os.path.dirname(weights_path), model_path, loaded_iter)\n')
    p.write_text(s,encoding='utf-8')
    assert runtime_hashes(source)==parent
    write_json(receipt,{'parent':parent,'fork':runtime_hashes(TARGET)})
    return TARGET
