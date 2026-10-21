"""Frozen independent PIDG fork with disabled-by-default metric tuning hooks."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from pidg_benchmark import read_json, runtime_hashes, write_json, sha256

SOURCE = ROOT / 'third_party/PIDG'
TARGET = Path(__file__).parent / 'code'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unrecognized PIDG patch anchor: ' + old[:100])
    return text.replace(old, new, 1)


def prepare():
    receipt = TARGET.parent / 'source_manifest.json'
    original = runtime_hashes(SOURCE)
    if TARGET.exists():
        saved = read_json(receipt)
        if saved['original'] != original or saved['fork'] != runtime_hashes(TARGET):
            raise ValueError('Existing tuning fork or original baseline changed')
        return TARGET
    for relative in original:
        destination = TARGET / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / relative, destination)
    path = TARGET / 'scene_PIDG/network.py'
    text = path.read_text(encoding='utf-8')
    text = replace_once(text, '            directional=True,', '            directional=True,\n            metric_residual_width=0,')
    text = replace_once(text, '        self.grid_mlp = nn.Sequential(*mlp)',
                        '        self.grid_mlp = nn.Sequential(*mlp)\n'
                        '        self.metric_residual = None\n'
                        '        if metric_residual_width:\n'
                        '            from variants.metric_tuning.model import ResidualAdapter\n'
                        '            self.metric_residual = ResidualAdapter(width, metric_residual_width)')
    text = replace_once(text, '        h = self.grid_mlp(h)',
                        '        h = self.grid_mlp(h)\n'
                        '        if self.metric_residual is not None:\n'
                        '            h = self.metric_residual(h)')
    path.write_text(text, encoding='utf-8')
    path = TARGET / 'train_pidg.py'
    text = path.read_text(encoding='utf-8')
    text = replace_once(text, '    from training_checkpoint import restore_training_state, save_training_state',
                        '    from variants.metric_tuning.checkpoint import restore_training_state, save_training_state')
    text = replace_once(text, '    gaussians = GaussianModel(dataset.sh_degree)',
                        '    from variants.metric_tuning.runtime import Tuning\n'
                        '    tuning = Tuning(os.environ.get("PIDG_METRIC_CONFIG"), args.model_path)\n'
                        '    gaussians = GaussianModel(dataset.sh_degree)')
    text = replace_once(text, '    scene = Scene(dataset, gaussians)',
                        '    scene = Scene(dataset, gaussians, load_iteration="best") if tuning.warm_start else Scene(dataset, gaussians)\n'
                        '    tuning.after_scene(gaussians, deform)')
    text = replace_once(text, 'restore_training_state(resume_checkpoint, gaussians, deform, opt, scene)',
                        'restore_training_state(resume_checkpoint, gaussians, deform, opt, scene, tuning)')
    text = replace_once(text, '    progress_bar = tqdm(',
                        '    if tuning.enabled and not resume_checkpoint:\n'
                        '        with torch.no_grad():\n'
                        '            tuning.evaluate(0, scene, deform, render, pipe, background, dataset.load2gpu_on_the_fly)\n'
                        '        best_psnr = tuning.best["PSNR"]["value"]\n'
                        '        best_iteration = 0\n'
                        '    progress_bar = tqdm(')
    old = '        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + reg'
    text = replace_once(text, old,
                        '        if tuning.enabled:\n'
                        '            loss = tuning.image_loss(iteration, image, gt_image, opt) + reg\n'
                        '        else:\n' + '    ' + old)
    old = ('            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss,\n'
           '                                       testing_iterations, scene, render, (pipe, background), deform,\n'
           '                                       dataset.load2gpu_on_the_fly)')
    text = replace_once(text, old,
                        '            if tuning.enabled:\n'
                        '                cur_psnr = None\n'
                        '                if iteration in testing_iterations:\n'
                        '                    cur_psnr = tuning.evaluate(iteration, scene, deform, render, pipe, background, dataset.load2gpu_on_the_fly)\n'
                        '                    best_psnr = tuning.best["PSNR"]["value"]\n'
                        '                    best_iteration = tuning.best["PSNR"]["iteration"]\n'
                        '            else:\n' + '\n'.join('    ' + line for line in old.splitlines()))
    text = replace_once(text, '            if iteration in testing_iterations:',
                        '            if iteration in testing_iterations and not tuning.enabled:') if text.count('            if iteration in testing_iterations:') == 1 else text
    # Distinguish the original best saver from the nested tuning evaluation.
    text = replace_once(text, '            if iteration in testing_iterations:\n                if cur_psnr.item()',
                        '            if iteration in testing_iterations and not tuning.enabled:\n                if cur_psnr.item()') if '            if iteration in testing_iterations:\n                if cur_psnr.item()' in text else text
    text = replace_once(text, '            checkpoint_path = os.path.join(args.model_path, "checkpoints", "resume.pt")',
                        '            tuning.request_stop(iteration)\n'
                        '            checkpoint_path = os.path.join(args.model_path, "checkpoints", "resume.pt")')
    path.write_text(text, encoding='utf-8')
    path = TARGET / 'hashencoder/backend.py'
    text = path.read_text(encoding='utf-8')
    text = replace_once(text, '_src_path = os.path.dirname(os.path.abspath(__file__))',
                        "_src_path = os.environ.get('PIDG_HASH_SOURCE_DIR', os.path.dirname(os.path.abspath(__file__)))")
    text = replace_once(text, "build_directory='./tmp_build/'", "build_directory=os.environ.get('PIDG_HASH_BUILD_DIR', './tmp_build/')")
    path.write_text(text, encoding='utf-8')
    assert runtime_hashes(SOURCE) == original
    write_json(receipt, {'original': original, 'fork': runtime_hashes(TARGET)})
    return TARGET


if __name__ == '__main__':
    print(prepare())
