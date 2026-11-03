"""Frozen-I7 inference intervention; never presented as a newly trained model."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch

from scripts.audit_invertible_strain import ROOT, EXP, CHECKPOINT, EXPECTED_SHA, digest


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-read-only-intervention', action='store_true')
    args = parser.parse_args()
    if not args.run_read_only_intervention:
        raise SystemExit('Frozen-model intervention is opt-in')
    out = ROOT/'output/staged_reconstruction/invertible_motion/covariance_intervention'
    assert not (out/'receipt.json').exists()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    from variants.staged_reconstruction.early_depth.train import load_initial
    from variants.staged_reconstruction.invertible_motion.core import install
    from variants.staged_reconstruction.invertible_motion.freeze import verify
    from variants.staged_reconstruction.train_static import camera, PIPE
    from variants.staged_reconstruction.stage2.model import compose
    from variants.staged_reconstruction.prepare_data import DATA, read, write
    verify()
    assert digest(CHECKPOINT) == EXPECTED_SHA
    renderer, scene, opt, initial, _ = load_initial()
    del opt, initial
    install(scene, read(EXP/'protocol.json'), True)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    scene.load_state_dict(state['model']); del state
    meta = read(DATA/'transforms_train.json')
    poses = {int(Path(frame['file_path']).stem): frame for frame in meta['frames']}
    panels, rows = [], []
    def pixels(value):
        return (value.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
    for cam_id, number in [(188, 43), (188, 146), (86, 86)]:
        cam = camera(poses[cam_id], meta, 1600)
        dynamic = scene.dynamic_at((number-1)/199)
        raw_norm = dynamic.get_rotation.norm(dim=-1)
        assert raw_norm.min() > 1e-6
        normalized = SimpleNamespace(**dict(vars(dynamic), get_rotation=torch.nn.functional.normalize(dynamic.get_rotation, dim=-1)))
        images, masks = [], []
        for label, value in [('I7_original', dynamic), ('I7_inference_only_unit_quaternion', normalized)]:
            package = renderer(cam, compose(scene.static, [value]), PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
            dyn = renderer(cam, value, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
            rgb = Image.fromarray(pixels(package['render']))
            alpha = (dyn['alpha'].squeeze().cpu().numpy().clip(0, 1)*255).round().astype(np.uint8)
            rgb.save(out/f'c{cam_id}_t{number}_{label}_native.png')
            Image.fromarray(alpha).save(out/f'c{cam_id}_t{number}_{label}_alpha.png')
            images.append(rgb); masks.append(alpha)
        ys, xs = np.nonzero(np.maximum(*masks) > 12)
        box = (max(0, int(xs.min())-30), max(0, int(ys.min())-30), min(1600, int(xs.max())+31), min(900, int(ys.max())+31)) if len(xs) else (0, 0, 1600, 900)
        sheet = Image.new('RGB', (1200, 400), '#171a22')
        draw = ImageDraw.Draw(sheet)
        for k, (label, im) in enumerate(zip(['I7 original frozen checkpoint', 'Same checkpoint: normalize quaternion at inference'], images)):
            crop = im.crop(box); crop.thumbnail((600, 365))
            sheet.paste(crop, (600*k, 30))
            draw.text((600*k+5, 8), f'c{cam_id}/t{number} | {label}', fill='white')
        sheet.save(out/f'c{cam_id}_t{number}_comparison.png')
        panels.append(sheet)
        rows.append({'camera': cam_id, 'frame': number, 'common_crop': box, 'raw_quaternion_min_norm': float(raw_norm.min())})
        print('COVARIANCE_INTERVENTION', cam_id, number, flush=True)
    contact = Image.new('RGB', (1200, 400*len(panels)))
    for k, panel in enumerate(panels):
        contact.paste(panel, (0, k*400))
    contact.save(out/'comparison.png')
    assert digest(CHECKPOINT) == EXPECTED_SHA
    receipt = {'checkpoint_sha256': EXPECTED_SHA, 'script_sha256': digest(Path(__file__)), 'optimizer_updates': 0,
               'new_trained_model': False, 'RGB_read': False, 'native_width': 1600, 'probes': rows,
               'scope': 'Inference-only covariance intervention on failed frozen I7 warmup. Same centers, scales, opacity, appearance and original renderer; only dynamic covariance quaternion is normalized. No full acceptance, causal proof, physical-GT claim or test evaluation.'}
    write(out/'receipt.json', receipt)


if __name__ == '__main__':
    main()
