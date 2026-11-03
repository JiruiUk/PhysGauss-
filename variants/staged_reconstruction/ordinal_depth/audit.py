"""Read-only training-depth qualification, including native backward verification."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
import torch
from ..prepare_data import ROOT, REPORT, DATA, read, write, sha
from ..train_static import camera, PIPE
from ..stage2.train import image
from ..union.core import dynamic_mask
from ..motion_integrity.prepare import load_parent, PARENT, PARENT_SHA
from ..motion_integrity.evaluate import pixels, tile

EXP = REPORT/'ordinal_depth'
OUT = ROOT/'output/staged_reconstruction/ordinal_depth'


def heat(value, mask):
    q = np.quantile(value[mask], [.02, .98])
    norm = np.clip((value-q[0])/max(q[1]-q[0], 1e-6), 0, 1)
    rgb = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    rgb[~mask] = 0
    return rgb


def main():
    torch.set_num_threads(4)
    renderer, scene, _, _, _ = load_parent()
    meta = read(DATA/'transforms_train.json')
    frames = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    rows, panels = [], []
    for number in [2, 43, 86, 146, 188, 200]:
        frame = frames[number]
        cam = camera(frame, meta, 800)
        rgba = image(frame, 800)
        dm = dynamic_mask(rgba)
        r, g, b = rgba[..., :3].transpose(2, 0, 1)
        sm = (rgba[..., 3] > .95) & (b > r*1.3+.02) & (b > g*1.05)
        source = ROOT/f'data/PIDG/rubber_duck/depth_distill/{number:04d}.npy'
        raw = np.load(source)
        pseudo = cv2.resize(raw, (800, 450), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            package = scene.render(renderer, cam, frame['time'])
            alpha = package['alpha'].squeeze().cpu().numpy()
            dep = package['depth'].squeeze().cpu().numpy()
            normal = dep/np.maximum(alpha, 1e-6)
            known = sm & (alpha > .99) & np.isfinite(normal)
            valid = (rgba[..., 3] > .95) & (alpha > .99)
        # No GT used: agreement with an existing frozen static model is only
        # a direction/quality diagnostic, not independent geometric validation.
        rows.append({'frame': number, 'source_sha256': sha(source), 'source_shape': list(raw.shape),
                     'finite': bool(np.isfinite(raw).all()), 'static_pixels': int(known.sum()),
                     'static_pseudo_vs_normalized_render_spearman': float(spearmanr(pseudo[known], normal[known]).statistic),
                     'dynamic_pseudo_vs_normalized_render_spearman': float(spearmanr(pseudo[dm], normal[dm]).statistic),
                     'pseudo_dynamic_quantiles': np.quantile(pseudo[dm], [.01,.1,.5,.9,.99]).tolist(),
                     'render_dynamic_quantiles': np.quantile(normal[dm], [.01,.1,.5,.9,.99]).tolist(),
                     'raw_over_alpha_median': float(np.median(dep[valid]/normal[valid]))})
        panel = [tile(pixels(package['render']), f'parent c/t{number}'),
                 tile((rgba[..., :3]*255).astype(np.uint8), f'train RGB {number}'),
                 tile(heat(pseudo, rgba[...,3]>.95), 'cached learned pseudo-depth'),
                 tile(heat(normal, valid), 'parent depth / alpha')]
        panels.append(np.concatenate(panel, axis=1))
        # Actual loss-to-geometry backward check on the dynamic-only render.
        dynamic = scene.dynamic_at(frame['time'])
        package = renderer(cam, dynamic, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
        normalized = package['depth'].squeeze()/package['alpha'].squeeze().clamp_min(1e-6)
        selected = torch.tensor(dm, device='cuda')
        loss = normalized[selected].mean()
        variables = [scene.pools[0]._xyz, *scene.grids.parameters(), *scene.networks.parameters()]
        gradients = torch.autograd.grad(loss, variables, allow_unused=True)
        rows[-1]['native_depth_gradient'] = {'finite': all(g is None or g.isfinite().all().item() for g in gradients),
            'canonical_norm': float(gradients[0].norm()),
            'field_norm': sum(float(g.norm()) for g in gradients[1:] if g is not None)}
        print('DEPTH_QUALIFICATION', rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(panels, axis=0)).save(OUT/'depth_qualification.png')
    write(EXP/'data_audit.json', {'training_only': True, 'depth_is_GT': False, 'optimizer_updates': 0,
          'parent_unchanged': sha(PARENT) == PARENT_SHA, 'rows': rows,
          'static_comparison_is_model_conditioned': True, 'visual_review': 'pending'})


if __name__ == '__main__':
    main()
