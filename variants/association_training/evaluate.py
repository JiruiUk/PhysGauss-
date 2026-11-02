"""Evaluate fixed full states, complete native contributions and uncertain IDs."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import itertools
import json
import os
import runpy
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.association_training.run import RUNS, REPORT, SOURCE, NAMES
from variants.metric_tuning.runtime import read_json, write_json, sha256

OUT = ROOT / 'output/association_training'
TRAIN = ROOT / 'data/derived/kinematic_resplit/training'
SELECTED = [2, 14, 27, 32, 42, 43, 55, 63, 74, 86, 108, 134, 172, 188, 200]
PALETTE = np.array([[236, 72, 153], [6, 182, 212], [34, 197, 94], [245, 158, 11]]) / 255


def load(directory):
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.deform_model import DeformModel
    checkpoint = directory / 'checkpoints/resume.pt'
    s = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert read_json(checkpoint.with_suffix('.json'))['sha256'] == sha256(checkpoint)
    settings = runpy.run_path(str(directory / 'model_config.py'))
    g = GaussianModel(3)
    for name, value in s['gaussian_parameters'].items():
        setattr(g, name, torch.nn.Parameter(value.cuda(), requires_grad=False))
    for name, value in s['gaussian_buffers'].items():
        setattr(g, name, value.cuda())
    for name, value in s['gaussian_scalars'].items():
        setattr(g, name, value)
    g.velocity_net.load_state_dict(s['velocity_net'])
    g.velocity_net.requires_grad_(False)
    d = DeformModel(settings['grid_args'], settings['network_args'], scale_xyz=settings.get('scale_xyz', 1.),
                    reg_spatial_able=False, reg_temporal_able=False)
    d.grid4d.load_state_dict(s['grid4d'])
    d.deform.load_state_dict(s['deform'])
    d.grid4d.requires_grad_(False); d.deform.requires_grad_(False)
    for key in ['gaussian_parameters', 'velocity_net', 'grid4d', 'deform']:
        assert all(v.isfinite().all() for v in s[key].values())
    return g, d, {'iteration': s['iteration'], 'sha256': sha256(checkpoint), 'point_count': len(g.get_xyz)}


def rgb(frame, width=1600):
    with Image.open(TRAIN / (frame['file_path'] + '.png')) as im:
        rgba = np.asarray(im.convert('RGBA')) / 255
    value = (rgba[..., :3] * rgba[..., 3:4] * 255).astype('uint8')
    return np.array(Image.fromarray(value).resize((width, width * 9 // 16)))


def render_call(cam, g, warp, color=None):
    from gaussian_renderer import render
    from variants.shape_coherence.observations import PIPE
    return render(cam, g, PIPE, torch.zeros(3, device='cuda'), warp['d_xyz'], warp['d_rotation'], warp['d_scaling'], override_color=color)


def complete_mass(cam, g, warp, masks, known):
    values = []
    errors = []
    alpha = None
    for start in [0, 3]:
        with torch.enable_grad():
            color = torch.ones(len(g.get_xyz), 3, device='cuda', requires_grad=True)
            package = render_call(cam, g, warp, color)
            alpha = package['alpha'].squeeze().detach()
            assert (package['render'].detach() - alpha[None]).abs().max() < 1e-6
            target = masks[start:start + 3]
            grad = torch.autograd.grad((package['render'] * target).sum(), color)[0].detach()
            expected = (alpha[None] * target).sum((1, 2))
            error = float(((grad.sum(0) - expected).abs() / expected.clamp_min(1)).max())
            assert error < 5e-4 and grad.isfinite().all() and grad.min() >= -1e-7
            values.append(grad[known].cpu().numpy() / 16)
            errors.append(error)
    return np.concatenate(values, axis=1), alpha, max(errors)


def candidate(matrix, center_fraction):
    """Compatibility only: no physical identity is inferred from a model's own map."""
    possibilities = list(itertools.permutations(range(4)))
    scores = np.array([sum(matrix[k, p[k]] for k in range(4)) for p in possibilities])
    order = np.argsort(-scores, kind='stable')
    best = possibilities[order[0]]
    active = matrix.sum(1) >= .2
    purity = [float(matrix[k, best[k]] / max(matrix[k].sum(), 1e-20)) for k in range(4)]
    # Even a clean permutation remains only a conditional model candidate.
    return {'best_component_permutation': list(best), 'score_mass': float(scores[order[0]]),
            'runner_up_gap_mass': float(scores[order[0]] - scores[order[1]]), 'active_groups': active.tolist(),
            'per_group_duck_region_purity': purity,
            'all_four_groups_supported': bool(active.all() and min(purity) >= .95 and all(center_fraction[k, best[k]] >= .5 for k in range(4))),
            'physical_ID_confirmed': False}


def evaluate_model(name, directory, metadata, components, frames, labels, image_mapping):
    from variants.shape_coherence.observations import camera
    from variants.flow_comparison.temporal import warp_at
    from variants.kinematic_tuning.model import center_trajectory
    from variants.flow_comparison.flow import projected
    from output.surface_correspondence.audit import center_inside
    from variants.surface_correspondence.model import particle_support
    os.environ['PIDG_METRIC_CONFIG'] = str(directory / 'metric_config.json')
    from variants.color_repair.color import _read
    _read.cache_clear()
    g, d, checkpoint = load(directory)
    known = torch.tensor(np.flatnonzero(labels >= 0), device='cuda')
    local_labels = labels[labels >= 0]
    model_out = OUT / name
    model_out.mkdir(parents=True, exist_ok=True)
    store = RUNS / 'evaluation_cache' / name
    store.mkdir(parents=True, exist_ok=True)
    rows = []
    all_mass = []; all_means = []; all_cov = []; all_depth = []
    for index, number in enumerate(frames):
        frame = metadata['frames'][index]
        assert int(Path(frame['file_path']).stem) == number
        cam = camera(frame, metadata, 1600)
        with torch.no_grad():
            w = warp_at(d, g.get_xyz.detach(), (number - 1) / 199)
            centers = center_trajectory(g.get_xyz.detach(), w['d_xyz'])
            smallcam = camera(frame, metadata, 400)
            means, cov = projected(centers[known], (g.get_scaling + w['d_scaling'])[known],
                                   (g.get_rotation + w['d_rotation'])[known], smallcam)
            view = smallcam.world_view_transform
            depth = (centers[known] @ view[:3, :3] + view[3, :3])[:, 2]
        small_masks = np.stack([components[number] >= 0, np.ones((225, 400), bool),
                                *(components[number] == k for k in range(4))])
        masks = torch.tensor(np.repeat(np.repeat(small_masks, 4, axis=1), 4, axis=2), device='cuda', dtype=torch.float32)
        mass, alpha, error = complete_mass(cam, g, w, masks, known)
        means = means.cpu().numpy(); cov = cov.cpu().numpy(); depth = depth.cpu().numpy()
        total_duck = float((alpha * masks[0]).sum()) / 16
        group_matrix = np.stack([mass[local_labels == k, 2:6].sum(axis=0, dtype=np.float64) for k in range(4)])
        center_fraction = np.zeros((4, 4))
        correct_points = 0
        known_own_mass = 0.
        confirmed_groups = 0
        for seed in range(4):
            for region in range(4):
                inside = center_inside(means[local_labels == seed], depth[local_labels == seed], cov[local_labels == seed], components[number] == region)
                center_fraction[seed, region] = float(inside.mean())
            region = int(image_mapping[index, seed])
            if region >= 0:
                ids = local_labels == seed
                inside = center_inside(means[ids], depth[ids], cov[ids], components[number] == region)
                good = particle_support(mass[ids, region + 2], mass[ids, 1], inside)
                correct_points += int(good.sum())
                known_own_mass += float(mass[ids, region + 2][good].sum(dtype=np.float64))
                confirmed_groups += 1
        duck_mass = float(mass[:, 0].sum(dtype=np.float64))
        all_image_mass = float(mass[:, 1].sum(dtype=np.float64))
        row = {'frame': int(number), 'early_bank_alpha_share_on_any_duck': duck_mass / max(total_duck, 1e-20),
               'early_bank_mass_on_duck': duck_mass, 'early_bank_total_image_mass': all_image_mass,
               'fraction_bank_mass_outside_ducks': 1 - duck_mass / max(all_image_mass, 1e-20),
               'image_IDs_still_independently_verified': confirmed_groups,
               'supported_points_on_independently_verified_IDs': correct_points,
               'supported_alpha_on_verified_IDs': known_own_mass / max(total_duck, 1e-20),
               'group_component_mass': group_matrix.tolist(), 'group_center_component_fraction': center_fraction.tolist(),
               'complete_contribution_relative_error': error,
               'candidate': candidate(group_matrix, center_fraction) if int(components[number].max()) == 3 else None,
               'native_total_duck_alpha_area': total_duck}
        rows.append(row)
        all_mass.append(mass); all_means.append(means); all_cov.append(cov); all_depth.append(depth)
        if number in SELECTED:
            with torch.no_grad():
                normal = render_call(cam, g, w)['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                color_images = []
                for offset in [0, 3]:
                    color = g.get_xyz.new_zeros(len(g.get_xyz), 3)
                    for k in range(min(3, 4 - offset)):
                        color[torch.tensor(labels == k + offset, device='cuda'), k] = 1
                    color_images.append(render_call(cam, g, w, color)['render'][:min(3, 4 - offset)].detach())
                contribution = torch.cat(color_images).permute(1, 2, 0).cpu().numpy() @ PALETTE
            pred = Image.fromarray((normal * 255).round().astype('uint8'))
            pred.resize((400, 225), Image.Resampling.LANCZOS).save(model_out / f'rgb{number:04d}.png')
            Image.fromarray((np.clip(contribution, 0, 1) * 255).round().astype('uint8')).resize((400, 225)).save(model_out / f'groups{number:04d}.png')
            panel = Image.fromarray(rgb(frame, 400))
            draw = ImageDraw.Draw(panel)
            for seed in range(4):
                for x, y in means[local_labels == seed]:
                    if np.isfinite([x, y]).all() and 0 <= x < 400 and 0 <= y < 225:
                        draw.ellipse((x - .55, y - .55, x + .55, y + .55), fill=tuple((PALETTE[seed] * 255).astype(int)))
            panel.save(model_out / f'centers{number:04d}.png')
            if number in [43, 74, 86, 200]:
                Image.fromarray(np.concatenate([rgb(frame), np.array(pred)], axis=1)).save(model_out / f'native_GT_Render{number:04d}.png')
        if (index + 1) % 15 == 0:
            print(name, 'support', index + 1, '/', len(frames), 'frame', number, row['early_bank_alpha_share_on_any_duck'], flush=True)
    np.savez_compressed(store / 'evidence.npz', frames=frames, labels=local_labels, kernel_ids=known.cpu().numpy(),
                        means=np.asarray(all_means), cov=np.asarray(all_cov), depth=np.asarray(all_depth),
                        mass=np.asarray(all_mass), image_mapping=image_mapping)
    del g, d
    torch.cuda.empty_cache()
    result = {'model': name, 'checkpoint': checkpoint, 'frames': rows, 'true_physical_ID_accuracy_available': False,
              'evaluator_sha256': sha256(Path(__file__)),
              'support_definition': 'Center and complete visible footprint; flow agreement is audited separately. Independently tracked image IDs terminate conservatively.',
              'cache_sha256': sha256(store / 'evidence.npz')}
    write_json(REPORT / (name + '_support.json'), result)
    return result


@torch.no_grad()
def validation(name, directory, metadata):
    from variants.shape_coherence.observations import camera
    from variants.flow_comparison.temporal import warp_at
    from variants.metric_tuning.model import quantize_png
    from utils.loss_utils import ssim
    from pytorch_msssim import ms_ssim
    import lpips
    os.environ['PIDG_METRIC_CONFIG'] = str(directory / 'metric_config.json')
    from variants.color_repair.color import _read
    _read.cache_clear()
    g, d, proof = load(directory)
    metric = lpips.LPIPS(net='vgg').cuda().eval().requires_grad_(False)
    rows = []
    for frame in metadata['frames']:
        number = int(Path(frame['file_path']).stem)
        cam = camera(frame, metadata, 1600)
        w = warp_at(d, g.get_xyz.detach(), (number - 1) / 199)
        pred = quantize_png(render_call(cam, g, w)['render'])[None]
        gt = torch.tensor(rgb(frame), device='cuda', dtype=torch.float32).permute(2, 0, 1)[None] / 255
        rows.append({'frame': number, 'PSNR': float(-10 * torch.log10((pred - gt).square().mean())),
                     'SSIM': float(ssim(pred, gt)), 'LPIPS': float(metric(pred * 2 - 1, gt * 2 - 1).mean()),
                     'MS-SSIM': float(ms_ssim(pred, gt, data_range=1., size_average=True))})
    result = {'model': name, 'full_state': proof, 'split': 'validation', 'count': len(rows), 'frames': rows,
              'evaluator_sha256': sha256(Path(__file__)),
              'metrics': {k: float(np.mean([v[k] for v in rows])) for k in ['PSNR', 'SSIM', 'LPIPS', 'MS-SSIM']}}
    write_json(REPORT / (name + '_validation.json'), result)
    print('FULL_STATE_VALIDATION', name, result['metrics'], flush=True)
    del g, d, metric
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['C1_original', *NAMES])
    args = parser.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / 'variants/association_training/code'))
    os.environ['PIDG_HASH_SOURCE_DIR'] = str(ROOT / 'third_party/PIDG/hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR'] = str(ROOT / 'third_party/PIDG/tmp_build')
    OUT.mkdir(parents=True, exist_ok=True)
    meta = read_json(TRAIN / 'transforms_train.json')
    meta['frames'].sort(key=lambda f: int(Path(f['file_path']).stem))
    frames = np.asarray([int(Path(f['file_path']).stem) for f in meta['frames']])
    with np.load(ROOT / 'output/surface_support/image_tracks.npz') as z:
        components = {int(n): c for n, c in zip(z['frames'], z['components'])}
    with np.load(ROOT / 'output/surface_correspondence/fragment_tracks.npz') as z:
        assert np.array_equal(z['frames'], frames)
        image_mapping = z['component_mapping'][0].copy()
    binding = read_json(REPORT / 'training_data.json')
    assert sha256(Path(binding['path'])) == binding['sha256']
    with np.load(binding['path']) as z:
        labels = z['labels'].copy()
    models = {'C1_original': SOURCE, **{n: RUNS / 'pilot' / n / 'rubber_duck' for n in NAMES}}
    write_json(REPORT / 'evaluation_plan_lock.json', {'models': {k: str(v) for k, v in models.items()},
               'selected_training_frames': SELECTED, 'all_training_frames_audited': frames.tolist(),
               'checkpoint_sha256': {k: sha256(v / 'checkpoints/resume.pt') for k, v in models.items()},
               'model_selection': 'fixed final full states; no metric tuning', 'source_sha256': sha256(Path(__file__))})
    for name, directory in models.items():
        if args.model and name != args.model:
            continue
        support_path = REPORT / (name + '_support.json')
        valid_path = REPORT / (name + '_validation.json')
        def reusable(path, field):
            if not path.exists():
                return False
            result = read_json(path)
            return result.get('evaluator_sha256') == sha256(Path(__file__)) and result[field]['sha256'] == sha256(directory / 'checkpoints/resume.pt')
        if not reusable(support_path, 'checkpoint'):
            evaluate_model(name, directory, meta, components, frames, labels, image_mapping)
        if not reusable(valid_path, 'full_state'):
            validation(name, directory, read_json(TRAIN / 'transforms_test.json'))
    print('ASSOCIATION EVALUATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
