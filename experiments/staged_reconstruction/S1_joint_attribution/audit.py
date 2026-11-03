"""Explicit opt-in read-only attribution. Disabled mode imports no experiment code."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_joint_attribution/v1'
OUT = ROOT/'output/staged_reconstruction/S1_joint_attribution/v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-joint-attribution', action='store_true')
    parser.add_argument('--phase', choices=['freeze', 'replay', 'measure', 'pipeline'], default='pipeline')
    parser.add_argument('--replicate', choices=['a', 'b'])
    args = parser.parse_args()
    if not args.enable_joint_attribution:
        print('S1_J2_DISABLED: no model imports or checkpoint reads'); return
    import os
    import sys
    import subprocess
    import traceback
    from datetime import datetime, timezone
    import numpy as np
    import torch
    from variants.staged_reconstruction.prepare_data import read, sha
    from variants.staged_reconstruction.joint_rgb import io
    from experiments.staged_reconstruction.S1_joint_attribution.core import (
        STATES, labels, mixed_payload, transitions, shift_sums, aggregate)
    io.RUNS = RUNS
    protocol = read(BASE/'protocol.json')
    accepted = ROOT/protocol['accepted_base']; accepted_runs = ROOT/protocol['accepted_runs']
    def stopped():
        paths = [ROOT/'stop.request', RUNS/'stop.request', RUNS.parent/'stop.request',
            RUNS.parent.parent/'stop.request', accepted_runs/'stop.request', accepted_runs.parent/'stop.request']
        return any(p.exists() for p in paths)
    def verify_manifest(manifest):
        for name, expected in manifest.items(): assert sha(ROOT/name) == expected, name
    def verify():
        assert not stopped(), 'User stop; do not resume'
        verify_manifest(read(BASE/'preregistration.json')['files'])
        verify_manifest(read(REPORT/'source_freeze.json'))
    if args.phase == 'freeze':
        assert not stopped() and not (REPORT/'source_freeze.json').exists()
        assert read(accepted/'v1/final_review.json')['status'] == 'completed_not_promoted'
        verify_manifest(read(BASE/'preregistration.json')['files'])
        files = read(accepted/'v1/source_freeze.json').copy(); verify_manifest(files)
        dataset = read(ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2/dataset.json')
        verify_manifest(dataset['files']); files.update(dataset['files'])
        files.update(read(BASE/'preregistration.json')['files'])
        paths = list(BASE.glob('*.py')) + [BASE/'preregistration.json', ROOT/'tests/test_joint_attribution.py']
        for path in paths: files[path.relative_to(ROOT).as_posix()] = sha(path)
        io.atomic_json(REPORT/'source_freeze.json', files)
        print('J2_SOURCE_DATA_FREEZE', len(files), flush=True); return
    verify()
    if args.phase == 'pipeline':
        assert not (REPORT/'measurement.json').exists(), 'Preserve completed measurement'
        def worker(phase, replicate=None):
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            path = REPORT/f'{stamp}_{phase}_{replicate}.log'
            command = [sys.executable, '-u', str(Path(__file__)), '--enable-joint-attribution', '--phase', phase]
            if replicate: command += ['--replicate', replicate]
            with path.open('x', encoding='utf-8') as log:
                code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            assert code == 0, f'Preserve failed log {path}'
        with io.lock('pipeline.lock'):
            state = {'status': 'running_read_only', 'pid': os.getpid(), 'training_updates': 0}
            io.atomic_json(RUNS/'state.json', state)
            try:
                for replicate in ['a', 'b']: worker('replay', replicate)
                for case in protocol['cases']:
                    with np.load(RUNS/f'{case}_a.npz') as a, np.load(RUNS/f'{case}_b.npz') as b:
                        assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
                io.atomic_json(REPORT/'native_replay.json', {'passed': True, 'independent_processes': 2,
                    'cases': protocol['cases'], 'camera': 24, 'all4_RGB_alpha_accumulated_depth_exact': True})
                worker('measure'); state['status'] = 'measured_pending_manual_review'
                io.atomic_json(RUNS/'state.json', state)
            except Exception:
                state.update(status='failed', traceback=traceback.format_exc())
                io.atomic_json(RUNS/'state.json', state); raise
        return
    torch.set_num_threads(4)
    from variants.staged_reconstruction.train_static import setup, render_static
    from variants.staged_reconstruction.adaptive_static.model import AdaptiveGaussian
    from variants.staged_reconstruction.nuisance_static.runtime import calibrated_camera
    renderer = setup()
    from PIL import Image, ImageDraw
    old_evaluation = read(accepted/'v1/synthetic_evaluation.json')
    old_protocol = read(accepted/'protocol.json')
    def load_models(case):
        payloads = []
        for arm in ['frozen_static', 'revisable_static']:
            state = torch.load(accepted_runs/'formal'/case/arm/'checkpoint.pt', map_location='cpu', weights_only=False)
            assert state['iteration'] == 6000
            payloads.append({k.removeprefix('static.'): v for k, v in state['model'].items() if k.startswith('static.')})
        p, c = payloads
        parent = torch.load(ROOT/old_protocol['parents'][case]['checkpoint'], map_location='cpu', weights_only=False)
        assert p.keys() == parent['model'].keys()
        assert all(torch.equal(p[k], parent['model'][k]) for k in p)
        models = {}
        for name in STATES:
            model = AdaptiveGaussian(ROOT/old_protocol['parents'][case]['initialization'])
            model.load_payload(mixed_payload(p, c, name)); model.requires_grad_(False); models[name] = model
        return models
    def render(models, camera):
        outputs = {}
        for name, model in models.items():
            package = render_static(renderer, camera, model)
            outputs[name] = {'rgb': package['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
                'alpha': package['alpha'].squeeze().cpu().numpy(), 'D': package['depth'].squeeze().cpu().numpy()}
        for first, second in [('pp', 'pc'), ('cp', 'cc')]:
            for key in ['alpha', 'D']: assert np.array_equal(outputs[first][key], outputs[second][key]), (first, second, key)
        return outputs
    def sheet(images, caption):
        height, width = images[0][1].shape[:2]
        canvas = Image.new('RGB', (width*2, (height+30)*2+34), '#141923')
        draw = ImageDraw.Draw(canvas); draw.text((4, 6), caption, fill='white')
        for i, (label, value) in enumerate(images):
            x, y = (i % 2)*width, 34+(i//2)*(height+30)
            draw.text((x+4, y+5), label, fill='white')
            canvas.paste(Image.fromarray((np.clip(value, 0, 1)*255).round().astype('uint8')), (x, y+30))
        return canvas
    results, images, invariance = {}, {}, {}
    with io.lock('read_only_native.lock'), torch.no_grad():
        for case in protocol['cases']:
            models = load_models(case)
            before = {s: {k: v.clone() for k, v in m.state_dict().items()} for s, m in models.items()}
            private = ROOT/protocol['data']/case/'evaluation_only'; meta = read(private/'cameras.json')
            frames = [r for r in meta['frames'] if r['time_index'] == 0]
            assert len(frames) == 32 and sum(not r['seen_camera'] for r in frames) == 8
            if args.phase == 'replay':
                assert args.replicate
                row = next(r for r in frames if r['camera'] == 24)
                camera = calibrated_camera(np.asarray(row['world_to_camera']), np.asarray(meta['K']), meta['width'], meta['height'])
                outputs = render(models, camera); path = RUNS/f'{case}_{args.replicate}.npz'; assert not path.exists()
                np.savez_compressed(path, **{f'{s}_{k}': v for s in STATES for k, v in outputs[s].items()})
                print('J2_REPLAY', case, args.replicate, sha(path), flush=True)
                continue
            assert args.phase == 'measure' and read(REPORT/'native_replay.json')['passed']
            OUT.mkdir(parents=True, exist_ok=True); rows = []
            for row in frames:
                assert not stopped(), 'User stop; leave partial outputs, no training state changed'
                camera = calibrated_camera(np.asarray(row['world_to_camera']), np.asarray(meta['K']), meta['width'], meta['height'])
                outputs = render(models, camera)
                with np.load(private/'gt'/f"c{row['camera']:02d}_t0.npz") as gt:
                    gt_depth = gt['static_depth'].copy(); gt_rgb = gt['static_rgb'].copy()
                    surface = gt_depth > 0; thin = gt['thin_static_visible'] & surface
                classes, depths, metrics = {}, {}, {}
                for name in STATES:
                    p = outputs[name]; classes[name], depths[name] = labels(p['alpha'], p['D'], gt_depth)
                    mse = float(np.mean((p['rgb']-gt_rgb)**2))
                    metrics[name] = {'MSE': mse, 'clean_GT_PSNR': float(-10*np.log10(max(mse, 1e-12))),
                        'static_RGB_spill': float(np.abs(p['rgb'][~surface]).mean())}
                record = {'camera': row['camera'], 'novel_camera': not row['seen_camera'], 'RGB': metrics,
                    'transitions': {g: transitions(classes['pp'], classes['cc'], mask) for g, mask in [('surface', surface), ('thin', thin)]},
                    'depth_shifts': {g: shift_sums(classes['pp'], classes['cc'], depths['pp'], depths['cc'], gt_depth, mask)
                        for g, mask in [('surface', surface), ('thin', thin)]}}
                original = next(r for r in old_evaluation['results'][case]['static_frames'] if r['camera'] == row['camera'])
                for name, arm in [('pp', 'frozen_static'), ('cc', 'revisable_static')]:
                    previous = original['models'][arm]; counts = previous['ray_counts']
                    for group, mask, prefix in [('surface', surface, 'static'), ('thin', thin, 'thin')]:
                        assert int(mask.sum()) == counts[f'{prefix}_GT_pixels']
                        assert int(((classes[name] != 2) & mask).sum()) == counts[f'{prefix}_valid_pixels']
                        assert int(((classes[name] == 0) & mask).sum()) == counts[f'accurate_{prefix}_ray_pixels']
                    for key in ['clean_GT_PSNR', 'static_RGB_spill']:
                        assert abs(metrics[name][key]-previous[key]) <= 1e-6, (case, row['camera'], name, key)
                rows.append(record)
                if row['camera'] in protocol['fixed_cameras']:
                    paths = [(OUT/f'{case}_c{row["camera"]:02d}_opacity_SH.png',
                        sheet([(s+' opacity='+s[0]+' SH='+s[1], outputs[s]['rgb']) for s in STATES], case+' fixed geometry; p parent / c candidate'))]
                    lost_bad = surface & (classes['pp'] == 0) & (classes['cc'] == 1)
                    lost_missing = surface & (classes['pp'] == 0) & (classes['cc'] == 2)
                    gained = surface & (classes['pp'] != 0) & (classes['cc'] == 0)
                    transition_rgb = np.full((*surface.shape, 3), .06, dtype=np.float32)
                    transition_rgb[surface] = [.2, .2, .2]
                    transition_rgb[lost_bad] = [1., .45, 0.]
                    transition_rgb[lost_missing] = [1., 0., 1.]
                    transition_rgb[gained] = [0., 1., .4]
                    paths.append((OUT/f'{case}_c{row["camera"]:02d}_ray_changes.png',
                        sheet([('Evaluator static GT', gt_rgb), ('Orange: lost but valid; pink: missing', transition_rgb),
                            ('Parent alpha', np.repeat(outputs['pp']['alpha'][..., None], 3, 2)),
                            ('Candidate alpha; green=gained above', np.repeat(outputs['cc']['alpha'][..., None], 3, 2))],
                            case+' c'+str(row['camera'])+' GT diagnostics only; gray=unchanged surface')))
                    for path, image in paths:
                        assert not path.exists(); image.save(path); images[path.relative_to(ROOT).as_posix()] = sha(path)
                print('J2_MEASURE', case, row['camera'], 'lost', record['transitions']['surface']['lost_accurate'], flush=True)
            results[case] = {'all32': aggregate(rows), 'novel8': aggregate([r for r in rows if r['novel_camera']]), 'frames': rows}
            for s, model in models.items():
                assert all(torch.equal(v, before[s][k]) for k, v in model.state_dict().items())
                assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            invariance[case] = {'model_values_and_geometry_exact': True, 'all_gradients_none': True,
                'SH_only_alpha_and_depth_exact_all32_cameras': True}
    if args.phase == 'replay': return
    verify()
    io.atomic_json(REPORT/'measurement.json', {'results': results, 'images': images,
        'invariance': invariance, 'source_freeze_sha256': sha(REPORT/'source_freeze.json'),
        'original_J1_A_per_camera_metrics_reproduced': True, 'training_updates': 0,
        'GT_evaluation_only': True, 'GT_pixel_arrays_or_training_assets_exported': False,
        'manual_review_pending': True})
    print('J2_COMPLETE_READ_ONLY_MEASUREMENT', flush=True)


if __name__ == '__main__': main()
