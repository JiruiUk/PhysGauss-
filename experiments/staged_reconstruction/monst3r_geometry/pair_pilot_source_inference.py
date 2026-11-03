"""Train-only MonST3R pair geometry pilot, separate from PIDG optimization."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from variants.staged_reconstruction.prepare_data import ROOT, DATA, read, write, sha

EXP = ROOT / 'experiments/staged_reconstruction/monst3r_geometry'
WORK = ROOT / 'runs/pidg_staged_reconstruction/monst3r_geometry/pilot'
ANCHORS = [2, 43, 86, 146, 188, 200]


def prepare():
    target = EXP / 'pilot_protocol.json'
    assert not target.exists()
    meta = read(DATA / 'transforms_train.json')
    ids = sorted(int(Path(f['file_path']).stem) for f in meta['frames'])
    partners = {str(n): sorted([v for v in ids if v != n], key=lambda v: (abs(v-n), v))[:2] for n in ANCHORS}
    write(target, {'anchors': ANCHORS, 'partners': partners, 'input_size': [512, 288],
        'RGB_source': 'training120 only, black alpha composite, official LANCZOS resizing and [-1,1] normalization',
        'global_optimizer_used': False, 'known_cameras_optimized': False,
        'reference': 'frozen S1 conditioned static depth, not GT',
        'alignment': 'one positive scalar per image pair from every third static-support pixel; no depth shift or affine inverse-depth fit',
        'why_pair_scale': 'Pair predictions have independent scale gauge; this differs from the VDA metric/global-scale protocol.',
        'held_static_median_max': .05, 'held_static_p90_max': .15, 'static_support_min': 1000,
        'dynamic_confidence_min': 3., 'dynamic_confident_fraction_min': .75,
        'dynamic_extrapolation_fraction_max': .1,
        'dynamic_partner_depth_median_max': .05, 'dynamic_partner_depth_p90_max': .15,
        'dynamic_self_projection_median_max_pixels': 2., 'dynamic_self_projection_p90_max_pixels': 5.,
        'source_manifest_sha256': sha(EXP / 'pair_vendor.json'),
        'weights_manifest_sha256': sha(EXP / 'asset_weights.json'),
        'test_or_val_RGB_used': False, 'PIDG_optimizer_updates': 0,
        'passing_is_not_dynamic_truth_or_full_training_admission': True})
    print('MONST3R_PAIR_PROTOCOL_FIXED', partners, flush=True)


def infer():
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    target = EXP / 'pair_inference.json'
    assert not target.exists()
    plan = read(EXP / 'pilot_protocol.json')
    assert sha(EXP / 'pair_vendor.json') == plan['source_manifest_sha256']
    assert sha(EXP / 'asset_weights.json') == plan['weights_manifest_sha256']
    lock = read(EXP / 'pair_vendor.json'); vendor = ROOT / lock['vendor_root']
    assert all(sha(vendor / name) == value for name, value in lock['files_sha256'].items())
    weights = read(EXP / 'asset_weights.json')
    path = ROOT / weights['weights_path']
    assert sha(path) == weights['weights_sha256']
    assert sha(path.parent / 'config.json') == weights['config_sha256']
    sys.path.insert(0, str(vendor))
    from dust3r.model import AsymmetricCroCo3DStereo
    torch.set_num_threads(4)
    config = read(path.parent / 'config.json')
    model = AsymmetricCroCo3DStereo(**config)
    state = load_file(path, device='cpu')
    # HF safetensors deduplicates eight tied DPT weights. Restore aliases
    # from its metadata, verifying that the live model really ties them.
    with safe_open(path, framework='pt', device='cpu') as tensor_file:
        aliases = tensor_file.metadata()
    live = model.state_dict()
    assert len(aliases) == 8 and set(live) - set(state) == set(aliases)
    for alias, canonical in aliases.items():
        assert canonical in state and alias in live
        assert live[alias].data_ptr() == live[canonical].data_ptr()
        state[alias] = state[canonical]
    model.load_state_dict(state, strict=True); del state
    model = model.eval().requires_grad_(False).cuda()
    meta = read(DATA / 'transforms_train.json')
    frames = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    WORK.mkdir(parents=True, exist_ok=True)
    receipts = []
    def view(n):
        source = DATA / (frames[n]['file_path'] + '.png')
        with Image.open(source) as im:
            rgba = np.array(im.convert('RGBA'))
        rgb = np.rint(rgba[..., :3] * (rgba[..., 3:4].astype(np.float32) / 255)).astype(np.uint8)
        im = Image.fromarray(rgb).resize(tuple(plan['input_size']), Image.Resampling.LANCZOS)
        tensor = torch.tensor(np.array(im), dtype=torch.float32, device='cuda').permute(2, 0, 1)[None] / 127.5 - 1
        return {'img': tensor, 'true_shape': torch.tensor([[288, 512]], device='cuda'), 'idx': n, 'instance': str(n)}, sha(source)
    with torch.inference_mode():
        for anchor in plan['anchors']:
            for partner in plan['partners'][str(anchor)]:
                first, hash_a = view(anchor); second, hash_b = view(partner)
                torch.cuda.reset_peak_memory_stats()
                # Official pair forward in float32; no flow or global optimizer.
                pred_a, pred_b = model(first, second)
                points = pred_a['pts3d'][0].cpu().numpy()
                confidence = pred_a['conf'][0].cpu().numpy()
                assert points.shape == (288, 512, 3) and confidence.shape == (288, 512)
                assert np.isfinite(points).all() and np.isfinite(confidence).all()
                dest = WORK / f'anchor{anchor:04d}_partner{partner:04d}.npz'
                assert not dest.exists()
                np.savez_compressed(dest, points=points, confidence=confidence)
                row = {'anchor': anchor, 'partner': partner, 'path': dest.relative_to(ROOT).as_posix(),
                       'sha256': sha(dest), 'anchor_RGB_sha256': hash_a, 'partner_RGB_sha256': hash_b,
                       'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
                receipts.append(row)
                print('MONST3R_PAIR_INFERRED', anchor, partner, float(confidence.min()), float(confidence.max()), row['peak_cuda_allocated_bytes'], flush=True)
                del first, second, pred_a, pred_b
    write(target, {'pairs': receipts, 'protocol_sha256': sha(EXP / 'pilot_protocol.json'),
                  'float32_official_pair_forward': True, 'strict_weights_load': True,
                  'safetensors_tied_weights_restored_from_verified_metadata': aliases,
                  'pilot_script_sha256': sha(Path(__file__)),
                  'global_optimizer_or_camera_optimization': False, 'PIDG_updates': 0,
                  'test_or_val_RGB_used': False, 'geometry_qualified_for_training': False})


def qualify():
    import cv2
    from scipy.ndimage import binary_erosion
    import torch
    from variants.staged_reconstruction.motion_integrity.prepare import load_parent
    from variants.staged_reconstruction.stage2.train import image
    from variants.staged_reconstruction.train_static import camera, PIPE
    from variants.staged_reconstruction.union.core import dynamic_mask
    from output.short_clip_diagnostic.camera_math import intrinsic
    target = EXP / 'pair_qualification.json'
    assert not target.exists()
    torch.set_num_threads(4)
    plan = read(EXP / 'pilot_protocol.json'); inference = read(EXP / 'pair_inference.json')
    assert inference['protocol_sha256'] == sha(EXP / 'pilot_protocol.json')
    meta = read(DATA / 'transforms_train.json')
    frames = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    renderer, scene, _, _, _ = load_parent()
    k = intrinsic(meta['camera_angle_x'], 'metadata', 512, 288)
    yy, xx = np.indices((288, 512))
    cache = {}; rows = []; context = {}
    with torch.no_grad():
        for row in inference['pairs']:
            anchor = row['anchor']; path = ROOT / row['path']; assert sha(path) == row['sha256']
            if anchor not in cache:
                rgba = image(frames[anchor], 512)
                r, g, b = rgba[..., :3].transpose(2, 0, 1)
                static = binary_erosion((rgba[..., 3] > .95) & (b > r*1.3+.02) & (b > g*1.05), iterations=3)
                package = renderer(camera(frames[anchor], meta, 512), scene.static, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
                alpha = package['alpha'].squeeze().cpu().numpy()
                reference = package['depth'].squeeze().cpu().numpy() / np.maximum(alpha, 1e-6)
                static &= (alpha > .99) & (reference > 1) & (reference < 15)
                dynamic = binary_erosion(dynamic_mask(rgba), iterations=2)
                cache[anchor] = rgba, static, dynamic, reference
            rgba, static, dynamic, reference = cache[anchor]
            with np.load(path) as data:
                points, confidence = data['points'], data['confidence']
            raw = points[..., 2]; finite = np.isfinite(raw) & (raw > 1e-5)
            support = static & finite & (confidence >= plan['dynamic_confidence_min'])
            y, x = np.nonzero(support)
            assert len(x) > 100, 'Insufficient scale support; no targets may be admitted'
            fit = np.arange(len(x)) % 3 == 0; held = ~fit
            scale = float(np.median(reference[y[fit], x[fit]] / raw[y[fit], x[fit]]))
            depth = raw * scale
            static_error = np.abs(depth[y[held], x[held]] - reference[y[held], x[held]])
            q = np.quantile(static_error, [.5, .9])
            dm = dynamic & finite & (confidence >= plan['dynamic_confidence_min'])
            fraction = float(dm.sum() / max(1, dynamic.sum()))
            lo, hi = np.quantile(raw[support], [.01, .99])
            extrap = float(((raw[dm] < lo) | (raw[dm] > hi)).mean()) if dm.any() else 1.
            uv = points @ k.T
            uv = uv[..., :2] / np.maximum(points[..., 2:3], 1e-6)
            error = np.linalg.norm(uv - np.stack([xx, yy], -1), axis=-1)
            projection = np.quantile(error[dm], [.5, .9]).tolist() if dm.any() else [float('inf')] * 2
            checks = {'static_support': len(x) >= plan['static_support_min'],
                      'held_static_median': q[0] <= plan['held_static_median_max'],
                      'held_static_p90': q[1] <= plan['held_static_p90_max'],
                      'dynamic_confidence': fraction >= plan['dynamic_confident_fraction_min'],
                      'dynamic_extrapolation': extrap <= plan['dynamic_extrapolation_fraction_max'],
                      'self_projection_median': projection[0] <= plan['dynamic_self_projection_median_max_pixels'],
                      'self_projection_p90': projection[1] <= plan['dynamic_self_projection_p90_max_pixels']}
            result = {'anchor': anchor, 'partner': row['partner'], 'scale': scale, 'static_support': len(x),
                      'static_error_median_p90': q.tolist(), 'dynamic_confident_fraction': fraction,
                      'dynamic_extrapolation_fraction': extrap, 'dynamic_projection_error_median_p90_pixels': projection,
                      'checks': checks, 'passed': all(checks.values())}
            rows.append(result); context.setdefault(anchor, []).append((depth, dm))
            print('MONST3R_PAIR_QUALIFIED', result, flush=True)
    stability = []
    for anchor, values in context.items():
        (a, mask_a), (b, mask_b) = values
        common = mask_a & mask_b
        q = np.quantile(np.abs(a[common] - b[common]), [.5, .9]).tolist() if common.any() else [float('inf')] * 2
        coverage = float(common.sum() / max(1, cache[anchor][2].sum()))
        passed = coverage >= plan['dynamic_confident_fraction_min'] and q[0] <= plan['dynamic_partner_depth_median_max'] and q[1] <= plan['dynamic_partner_depth_p90_max']
        stability.append({'anchor': anchor, 'common_dynamic_fraction': coverage, 'partner_depth_difference_median_p90': q, 'passed': passed})
    write(target, {'pair_rows': rows, 'partner_stability': stability,
        'pilot_checks_passed': all(r['passed'] for r in rows + stability),
        'protocol_sha256': sha(EXP / 'pilot_protocol.json'), 'inference_sha256': sha(EXP / 'pair_inference.json'),
        'static_reference_is_model_conditioned_not_GT': True, 'dynamic_geometry_truth_validated': False,
        'training_admitted': False, 'PIDG_updates': 0, 'test_or_val_RGB_used': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare', 'infer', 'qualify'])
    args = parser.parse_args()
    {'prepare': prepare, 'infer': infer, 'qualify': qualify}[args.operation]()
