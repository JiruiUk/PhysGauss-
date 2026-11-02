"""Independent checkpoint, cache and reported-mass checks (CPU)."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from variants.metric_tuning.runtime import read_json, write_json, sha256
from variants.association_training.freeze import freeze

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'experiments/association_training'
RUNS = ROOT / 'runs/pidg_association_training'
SOURCE = ROOT / 'runs/pidg_color_repair/C1_repaired/rubber_duck'
NAMES = ['T0_C1_continue', 'T1_association']


def finite_tree(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value, (list, tuple)):
        for v in value: finite_tree(v)


def main():
    torch.set_num_threads(4)
    frozen = freeze()
    source = torch.load(SOURCE / 'checkpoints/resume.pt', map_location='cpu', weights_only=False)
    original_sha = sha256(SOURCE / 'checkpoints/resume.pt')
    assert original_sha == read_json(REPORT / 'plan.json')['source_sha256']
    camera_lists = []
    branches = []
    checks = {}
    for name in NAMES:
        directory = RUNS / 'pilot' / name / 'rubber_duck'
        p = directory / 'checkpoints/resume.pt'
        assert sha256(p) == read_json(p.with_suffix('.json'))['sha256']
        state = torch.load(p, map_location='cpu', weights_only=False)
        assert state['iteration'] == 52000
        assert len(state['gaussian_parameters']['_xyz']) == 86143
        finite_tree({k: state[k] for k in ['gaussian_parameters', 'gaussian_buffers', 'gaussian_optimizer',
                                          'velocity_net', 'grid4d', 'deform', 'deform_optimizer']})
        step_deltas = {}
        for key in ['gaussian_optimizer', 'deform_optimizer']:
            before, after = source[key]['state'], state[key]['state']
            group_deltas = {}
            for group in source[key]['param_groups']:
                name_in_optimizer = group.get('name', 'unnamed')
                deltas = [int(after[k]['step']) - int(before[k]['step']) for k in group['params'] if k in before and 'step' in before[k]]
                expected = 20 if name_in_optimizer.startswith('velocity_net_') or name_in_optimizer == 'force_sigma' else 2000
                assert not deltas or set(deltas) == {expected}, (key, name_in_optimizer, set(deltas))
                group_deltas[name_in_optimizer] = sorted(set(deltas))
            step_deltas[key] = group_deltas
        camera_lists.append([json.loads(s) for s in (directory / 'camera_sampling.jsonl').read_text().splitlines()])
        branches.append(read_json(directory / 'branch.json'))
        checks[name] = {'sha256': sha256(p), 'finite_full_state_and_optimizers': True,
                        'optimizer_step_deltas': step_deltas, 'iteration': 52000, 'point_count': 86143}
        del state
    assert camera_lists[0] == camera_lists[1] and len(camera_lists[0]) == 2000
    assert [x['iteration'] for x in camera_lists[0]] == list(range(50001, 52001))
    assert branches[0]['identical_full_checkpoint_sha256'] == branches[1]['identical_full_checkpoint_sha256']
    configs = [read_json(RUNS / 'pilot' / name / 'rubber_duck/metric_config.json') for name in NAMES]
    assert {k for k in configs[0] if configs[0][k] != configs[1][k]} == {'association_training_enabled'}
    events = [json.loads(s) for s in (RUNS / 'pilot/T1_association/rubber_duck/association_loss.jsonl').read_text().splitlines()]
    assert len(events) == 100 and [r['iteration'] for r in events] == list(range(50020, 52001, 20))
    flow = read_json(ROOT / 'data/derived/flow_comparison_train_only/manifest.json')['pairs']
    for i, row in enumerate(events):
        pair = flow[(i * 37) % len(flow)]
        assert row['pair'] == [pair['source'], pair['target']]
        assert all(np.isfinite(row[k]) for k in ['leakage', 'coverage_floor', 'temporal', 'weighted_loss'])
    assert not (RUNS / 'pilot/T0_C1_continue/rubber_duck/association_loss.jsonl').exists()
    validation = {}
    original_cache_errors = []
    image_checks = 0
    native_psnr = {}
    train_root = ROOT / 'data/derived/kinematic_resplit/training'
    train_frames = {int(Path(f['file_path']).stem): f for f in read_json(train_root / 'transforms_train.json')['frames']}
    for name in ['C1_original', *NAMES]:
        report = read_json(REPORT / (name + '_support.json'))
        p = RUNS / 'evaluation_cache' / name / 'evidence.npz'
        assert sha256(p) == report['cache_sha256']
        z = np.load(p)
        frames, labels, mass = z['frames'], z['labels'], z['mass']
        assert mass.shape == (120, 2065, 6)
        assert np.isfinite(mass).all() and mass.min() >= -1e-7
        assert np.allclose(mass[:, :, 0], mass[:, :, 2:6].sum(-1), atol=2e-4, rtol=2e-4)
        assert (mass[:, :, 0] <= mass[:, :, 1] + 1e-4).all()
        for i, row in enumerate(report['frames']):
            assert int(frames[i]) == row['frame']
            matrix = np.stack([mass[i, labels == k, 2:6].sum(0, dtype=np.float64) for k in range(4)])
            assert np.allclose(matrix, row['group_component_mass'], atol=1e-5)
            share = mass[i, :, 0].sum(dtype=np.float64) / row['native_total_duck_alpha_area']
            assert abs(share - row['early_bank_alpha_share_on_any_duck']) < 1e-9
            if row['candidate'] is not None:
                assert row['candidate']['physical_ID_confirmed'] is False
            if name == 'C1_original':
                old = ROOT / f'runs/pidg_short_clip_diagnostic/cache/native{int(frames[i]):04d}.npz'
                if old.exists():
                    with np.load(old) as c:
                        reference = c['mass'][z['kernel_ids']]
                    assert np.allclose(mass[i, :, 2:6], reference[:, :4], atol=1e-3, rtol=1e-3)
                    # Historical native cache: duck0..3, BLUE, whole image.
                    assert np.allclose(mass[i, :, 1], reference[:, 5], atol=1e-3, rtol=1e-3)
                    original_cache_errors.append(float(np.max(np.abs(mass[i, :, 1] - reference[:, 5]))))
        v = read_json(REPORT / (name + '_validation.json'))
        assert v['count'] == 30
        for k in ['PSNR', 'SSIM', 'LPIPS', 'MS-SSIM']:
            assert abs(np.mean([r[k] for r in v['frames']]) - v['metrics'][k]) < 1e-10
        assert v['full_state']['sha256'] == report['checkpoint']['sha256']
        validation[name] = v['metrics']
        for number in [43, 74, 86, 200]:
            directory = ROOT / 'output/association_training' / name
            native = np.array(Image.open(directory / f'native_GT_Render{number:04d}.png'))
            assert native.shape == (900, 3200, 3)
            assert np.isfinite(native).all()
            with Image.open(train_root / (train_frames[number]['file_path'] + '.png')) as image:
                rgba = np.array(image.convert('RGBA')) / 255
            reference = np.array(Image.fromarray((rgba[..., :3] * rgba[..., 3:4] * 255).astype('uint8')).resize((1600, 900)))
            assert np.array_equal(native[:, :1600], reference), 'GT half differs from source frame'
            mse = np.mean(((native[:, 1600:].astype(np.float64) - reference) / 255) ** 2)
            native_psnr[f'{name}/{number}'] = float(-10 * np.log10(mse))
            image_checks += 1
        z.close()
    result = {'passed': True, 'frozen_files_verified': len(frozen['files']), 'historical_C1_unchanged': original_sha,
              'matched_camera_updates': 2000, 'association_optimizer_events': 100, 'final_checkpoints': checks,
              'full_contribution_reports_recomputed': 360, 'native_pair_images_checked': image_checks,
              'native_pair_GT_pixel_identity': True, 'selected_training_native_PSNR': native_psnr,
              'original_native_cache_frames_compared': len(original_cache_errors),
              'original_native_cache_max_mass_error': max(original_cache_errors),
              'validation_metrics': validation,
              'physical_identity_ground_truth_available': False}
    write_json(REPORT / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
