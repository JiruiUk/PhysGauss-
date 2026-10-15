"""Create independent PIDG code with a small optional extension hook."""
from pathlib import Path
import sys
import shutil
from pidg_benchmark import PROJECT,runtime_hashes,write_json


def prepare():
    source=PROJECT/'third_party/PIDG'
    target=PROJECT/'variants/constitutive_pidg/code'
    if target.exists():
        return target
    for rel in runtime_hashes(source):
        destination=target/rel;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/rel,destination)
    file=target/'train_pidg.py';text=file.read_text(encoding='utf-8')
    text=text.replace('    gaussians.training_setup(opt)\n','    gaussians.training_setup(opt)\n    from pidg_ext.runtime import Extension\n    extension = Extension(os.environ.get("PIDG_CONSTITUTIVE_CONFIG"), args.model_path, len(scene.getTrainCameras()))\n',1)
    assert text.count('        loss.backward()')==1
    text=text.replace('        loss.backward()', '        if extension.enabled:\n            extra_loss = extension.loss(iteration, gaussians, deform, d_xyz, fid, opt, cauchy_momentum_loss if iteration > opt.warm_up and iteration % 100 == 0 else None)\n            if extra_loss is not None:\n                loss = loss + extra_loss\n\n        loss.backward()',1)
    text=text.replace('                gaussians.optimizer.step()', '                extension.optimizer_step()\n                gaussians.optimizer.step()',1)
    text=text.replace('restore_training_state(resume_checkpoint, gaussians, deform, opt, scene)', 'restore_training_state(resume_checkpoint, gaussians, deform, opt, scene, extension)')
    file.write_text(text,encoding='utf-8')
    (target/'training_checkpoint.py').write_text('from pidg_ext.checkpoint import save_training_state, restore_training_state\n',encoding='utf-8')
    file=target/'hashencoder/backend.py';text=file.read_text(encoding='utf-8')
    text=text.replace("_src_path = os.path.dirname(os.path.abspath(__file__))", "_src_path = os.environ.get('PIDG_HASH_SOURCE_DIR', os.path.dirname(os.path.abspath(__file__)))")
    text=text.replace("build_directory='./tmp_build/'", "build_directory=os.environ.get('PIDG_HASH_BUILD_DIR', './tmp_build/')")
    file.write_text(text,encoding='utf-8')
    write_json(target.parent/'baseline_source_manifest.json',runtime_hashes(source))
    return target


if __name__=='__main__':print(prepare())
