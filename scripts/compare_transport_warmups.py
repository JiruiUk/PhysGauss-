"""Matched I8/I9 final warmup state and native image comparison."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from variants.staged_reconstruction.prepare_data import ROOT, REPORT, RUNS, read, write, sha
from scripts.compare_transport_previews import sampled_frames


def main():
    target = REPORT / 'relative_strain/comparison16000.json'
    assert not target.exists()
    torch.set_num_threads(4)
    branches = [('I8', 'covariance_transport', 'transport'), ('I9', 'relative_strain', 'strain')]
    initial, final, receipts = {}, {}, {}
    for label, branch, arm in branches:
        directory = RUNS / branch / 'formal' / arm
        receipt = read(directory / 'checkpoint.json')
        assert sha(directory / 'checkpoint.pt') == receipt['sha256']
        assert read(REPORT / branch / 'warmup_state_audit.json')['passed']
        final[label] = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=False)
        initial[label] = torch.load(directory / 'initial.pt', map_location='cpu', weights_only=False)
        assert final[label]['iteration'] == 16000
        receipts[label] = {'path': (directory / 'checkpoint.pt').relative_to(ROOT).as_posix(), 'sha256': receipt['sha256']}
    a, b = initial['I8'], initial['I9']
    assert a['model'].keys() == b['model'].keys() and all(torch.equal(v, b['model'][k]) for k, v in a['model'].items())
    assert a['optimizer'] == b['optimizer']
    a, b = final['I8'], final['I9']
    assert a['optimizer']['param_groups'] == b['optimizer']['param_groups']
    for key in ['camera_stack', 'rng_python']:
        assert a[key] == b[key]
    assert torch.equal(a['rng_torch'], b['rng_torch'])
    assert all(torch.equal(x, y) for x, y in zip(a['rng_cuda'], b['rng_cuda']))
    assert all(np.array_equal(x, y) for x, y in zip(a['rng_numpy'], b['rng_numpy']))
    assert sampled_frames('covariance_transport', 'transport', 16000) == sampled_frames('relative_strain', 'strain', 16000)
    out = ROOT / 'output/staged_reconstruction/relative_strain/compare_warmup16000'
    assert not out.exists(); out.mkdir(parents=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    panels = []; inputs = {}; crops = {}
    for cam, frame in [(188, 43), (188, 146), (86, 86)]:
        paths = [ROOT / f'output/staged_reconstruction/{branch}/warmup/fixed{cam}_t{frame:04d}_native.png' for _, branch, _ in branches]
        pictures = [Image.open(path).convert('RGB') for path in paths]
        assert all(im.size == (1600, 900) for im in pictures)
        support = np.maximum.reduce([np.asarray(im).max(-1) for im in pictures]) > 20
        y, x = np.nonzero(support)
        box = [max(0, int(x.min())-30), max(0, int(y.min())-30), min(1600, int(x.max())+31), min(900, int(y.max())+31)]
        key = f'c{cam}_t{frame}'; crops[key] = box
        height = round((box[3]-box[1])*800/(box[2]-box[0]))
        panel = Image.new('RGB', (1600, height+40), '#171a22')
        for column, ((label, _, _), path, im) in enumerate(zip(branches, paths, pictures)):
            panel.paste(im.crop(box).resize((800, height), Image.Resampling.LANCZOS), (column*800, 40))
            ImageDraw.Draw(panel).text((column*800+8, 8), f'{label} warmup16000 | c{cam}/t{frame}', font=font, fill='white')
            inputs[f'{label}/{key}'] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}
        panel.save(out / f'{key}.png'); panels.append(panel)
    sheet = Image.new('RGB', (1600, sum(im.height for im in panels)), '#171a22')
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y)); y += panel.height
    sheet.save(out / 'comparison.png')
    write(target, {'checkpoints': receipts, 'actual_warmup_updates_each': 16000, 'physics_calls_each': 0,
        'extended_initial_model_and_optimizer_exact': True,
        'same_optimizer_groups_sampler_resolution_and_Python_NumPy_CPU_CUDA_rng': True,
        'inputs': inputs, 'common_crop_per_row': crops, 'original_native_images_preserved': True,
        'outputs_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(out.glob('*.png'))},
        'script_sha256': sha(__file__), 'optimizer_updates_by_diagnostic': 0, 'test_RGB_read': False,
        'scope': 'One matched run per arm; both failed original static-preservation admission. No successful completeness claim.'})
    print('I8_I9_MATCHED_FINAL_WARMUP_COMPARISON', out, flush=True)


if __name__ == '__main__':
    main()
