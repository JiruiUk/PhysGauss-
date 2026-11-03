"""Report insufficient support without relaxing the preregistered threshold."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps

from variants.staged_reconstruction.prepare_data import ROOT, DATA, read, write, sha
from variants.staged_reconstruction.stage2.train import image
from variants.staged_reconstruction.union.core import dynamic_mask


def main():
    exp = ROOT / 'experiments/staged_reconstruction/monst3r_geometry'
    target = exp / 'confidence_audit.json'
    assert not target.exists()
    plan = read(exp / 'pilot_protocol.json'); inference = read(exp / 'pair_inference.json')
    assert inference['protocol_sha256'] == sha(exp / 'pilot_protocol.json')
    meta = read(DATA / 'transforms_train.json')
    frames = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    rows = []; panels = {}
    for receipt in inference['pairs']:
        path = ROOT / receipt['path']; assert sha(path) == receipt['sha256']
        with np.load(path) as values:
            points, confidence = values['points'], values['confidence']
        rgba = image(frames[receipt['anchor']], 512)
        r, g, b = rgba[..., :3].transpose(2, 0, 1)
        static = binary_erosion((rgba[..., 3] > .95) & (b > r*1.3+.02) & (b > g*1.05), iterations=3)
        dynamic = binary_erosion(dynamic_mask(rgba), iterations=2)
        valid = np.isfinite(points).all(-1) & (points[..., 2] > 1e-5) & (confidence >= plan['dynamic_confidence_min'])
        fraction = float((valid & dynamic).sum() / max(1, dynamic.sum()))
        rows.append({'anchor': receipt['anchor'], 'partner': receipt['partner'],
            'static_image_support': int(static.sum()), 'confident_static_support_upper_bound': int((valid & static).sum()),
            'dynamic_image_support': int(dynamic.sum()), 'dynamic_confident_fraction': fraction,
            'dynamic_confidence_q10_q50_q90_q99': np.quantile(confidence[dynamic], [.1, .5, .9, .99]).tolist(),
            'dynamic_confidence_gate_passed': fraction >= plan['dynamic_confident_fraction_min'],
            'static_support_gate_possible': int((valid & static).sum()) >= plan['static_support_min']})
        if receipt['anchor'] not in panels:
            rgb = np.rint(rgba[..., :3] * rgba[..., 3:4] * 255).clip(0, 255).astype(np.uint8)
            heat = np.rint(colormaps['magma'](np.clip((confidence - 1) / 4, 0, 1))[..., :3] * 255).astype(np.uint8)
            support = np.zeros_like(rgb)
            support[static] = [35, 60, 100]; support[dynamic] = [110, 95, 0]
            support[valid & static] = [30, 150, 255]; support[valid & dynamic] = [255, 230, 30]
            panels[receipt['anchor']] = (receipt['partner'], rgb, heat, support)
    out = ROOT / 'output/staged_reconstruction/monst3r_geometry'
    out.mkdir(parents=True, exist_ok=True)
    figure = out / 'adjacent_pair_confidence.png'; assert not figure.exists()
    sheet = Image.new('RGB', (1536, 6*318 + 44), '#141721')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), 'Train RGB | MonST3R confidence: black=1, white>=5 | Bright blue/yellow = static/dynamic support at fixed confidence>=3', fill='white')
    for index, anchor in enumerate(plan['anchors']):
        partner, *images = panels[anchor]
        top = 44 + index*318
        draw.text((8, top), f'anchor {anchor}, partner {partner}; masks from train RGB; no metric depth or 3D truth claim', fill='white')
        for column, panel in enumerate(images):
            sheet.paste(Image.fromarray(panel), (column*512, top+24))
    sheet.save(figure)
    impossible = any(not r['dynamic_confidence_gate_passed'] or not r['static_support_gate_possible'] for r in rows)
    assert impossible, 'If confidence gates are possible, finish the full geometry qualification instead'
    write(target, {'rows': rows, 'protocol_sha256': sha(exp / 'pilot_protocol.json'),
        'inference_sha256': sha(exp / 'pair_inference.json'), 'audit_script_sha256': sha(Path(__file__)),
        'adjacent_pair_pilot_rejected_by_necessary_support_conditions': True,
        'full_qualification_stopped_for_insufficient_scale_support': True,
        'figure': figure.relative_to(ROOT).as_posix(), 'figure_sha256': sha(figure),
        'static_counts_are_upper_bounds_before_frozen_reference_alpha_depth_filters': True,
        'prediction_confidence_is_not_probability_or_3D_truth': True,
        'geometry_targets_written_for_PIDG': False, 'PIDG_updates': 0,
        'gate_relaxed': False, 'whole_MonST3R_method_rejected': False})
    print('MONST3R_ADJACENT_CONFIDENCE_AUDIT', rows, flush=True)


if __name__ == '__main__':
    main()
