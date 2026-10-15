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
    text=text.replace('            cauchy_momentum_loss = cauchy_momentum_loss_sampled(', '            momentum_loss_fn = cauchy_momentum_loss_sampled\n            if extension.force_enabled:\n                extension.force_time, extension.force_iteration = fid, iteration\n                momentum_loss_fn = extension.momentum_loss\n            cauchy_momentum_loss = momentum_loss_fn(',1)
    text=text.replace('        # Every 1000 its', '        extension.before_iteration(iteration, gaussians, deform, opt)\n\n        # Every 1000 its',1)
    text=text.replace('            if extension.force_enabled:', '            if extension.force_enabled and extension.force_active:',1)
    text=text.replace('            if iteration < opt.densify_until_iter:', '            if iteration < opt.densify_until_iter and extension.densification_enabled:',1)
    text=text.replace('                gaussians.optimizer.step()', '                if extension.neural_step_enabled:\n                    gaussians.optimizer.step()',1)
    text=text.replace('                deform.optimizer.step()', '                if extension.neural_step_enabled:\n                    deform.optimizer.step()',1)
    file.write_text(text,encoding='utf-8')
    (target/'training_checkpoint.py').write_text('from pidg_ext.checkpoint import save_training_state, restore_training_state\n',encoding='utf-8')
    file=target/'hashencoder/backend.py';text=file.read_text(encoding='utf-8')
    text=text.replace("_src_path = os.path.dirname(os.path.abspath(__file__))", "_src_path = os.environ.get('PIDG_HASH_SOURCE_DIR', os.path.dirname(os.path.abspath(__file__)))")
    text=text.replace("build_directory='./tmp_build/'", "build_directory=os.environ.get('PIDG_HASH_BUILD_DIR', './tmp_build/')")
    file.write_text(text,encoding='utf-8')
    write_json(target.parent/'baseline_source_manifest.json',runtime_hashes(source))
    return target


if __name__=='__main__':print(prepare())
