"""Read-only S3-U reliability audit on real train images with pinned inputs."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from PIL import Image, ImageDraw
from ..prepare_data import ROOT, DATA, read, write, sha
from ..train_static import camera, PIPE
from ..stage2.model import compose
from ..stage2.train import image
from ..union.core import dynamic_mask
from ..motion_integrity.prepare import PARENT, PARENT_SHA, load_parent
from .native_audit import EXP, OUT, detached_bank, support_render, snapshot_rng, assert_rng, NAMES
from .core import color_vjp, squared_support, recurring_candidates


def model_digest(scene):
    result = hashlib.sha256()
    for name, value in sorted(scene.state_dict().items()):
        result.update(name.encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def independent_agreement(a, b, qualified, plan):
    active = (a+b)*.5 > plan['support_floor']
    if not active.any():
        return {'active_points': 0, 'qualified_points': int(qualified.sum()),
            'passed': False, 'reason': 'No positive above-floor compositing support'}
    relative = ((a-b).abs()/((a+b)*.5).clamp_min(1e-12))[active]
    n = a.numel(); k = max(1, int(n*plan['top_fraction']))
    def top(value):
        order = torch.argsort(torch.where(qualified, value, torch.zeros_like(value)), descending=True, stable=True)
        return set(order[:k][qualified[order[:k]] & (value[order[:k]] > 0)].cpu().tolist())
    x, y = top(a), top(b)
    overlap = len(x & y)/max(1, min(len(x), len(y))) if x and y else 0.
    rho = float(spearmanr(a[active].cpu().numpy(), b[active].cpu().numpy()).statistic)
    median = float(relative.median()); p90 = float(torch.quantile(relative, .9))
    passed = (median <= plan['seed_relative_median_max'] and p90 <= plan['seed_relative_p90_max']
        and rho >= plan['seed_spearman_min'] and overlap >= plan['seed_top_overlap_min'])
    return {'active_points': int(active.sum()), 'qualified_points': int(qualified.sum()),
        'relative_difference_median': median, 'relative_difference_p90': p90,
        'spearman': rho, 'qualified_top2_overlap': overlap, 'passed': passed}


def pixels(value):
    return (value.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)


def measure(renderer, scene, frame, meta, width, plan):
    cam = camera(frame, meta, width)
    rgba = image(frame, width)
    gt = torch.tensor(rgba[..., :3]*rgba[..., 3:4], device='cuda').permute(2, 0, 1)
    with torch.no_grad():
        bank = detached_bank(compose(scene.static, [scene.dynamic_at(frame['time'])]))
        reference = scene.render(renderer, cam, frame['time'])
        direct = renderer(cam, bank, PIPE, gt.new_zeros(3), 0., 0., 0.)
        assert torch.equal(reference['render'], direct['render'])
        assert torch.equal(reference['alpha'], direct['alpha'])
        pred = direct['render'].detach().clone()
        foreground = torch.tensor(dynamic_mask(rgba), device='cuda')
        error = (pred-gt).abs().mean(0)
        good = foreground & (error < plan['photometric_MAE_max'])
        del reference, direct
    package, colors = support_render(renderer, cam, bank)
    rgb = package['render']
    mass = color_vjp(rgb, colors, torch.ones_like(rgb[0])).double()
    good_mass = color_vjp(rgb, colors, good.to(rgb.dtype)).double()
    good_fraction = torch.where(mass > 0, good_mass/mass.clamp_min(1e-30), torch.zeros_like(mass))
    frame_id = int(Path(frame['file_path']).stem)
    estimates = [squared_support(rgb, colors, batches=plan['batches'], seed=seed+1009*frame_id+width)['support']
                 for seed in plan['seeds']]
    offset = len(scene.static.get_xyz)
    qualified = ((good_fraction[offset:] >= plan['converged_foreground_mass_fraction_min'])
        & (bank.get_opacity[offset:, 0] > plan['opacity_min'])
        & ((estimates[0][offset:]+estimates[1][offset:])*.5 >= plan['support_floor']))
    agreement = independent_agreement(estimates[0][offset:], estimates[1][offset:], qualified, plan)
    del package, rgb, colors
    torch.cuda.empty_cache()
    # Deterministic population-spanning rows, independent of estimated rank.
    row_indices = np.unique(np.linspace(0, len(qualified)-1, plan['exact_sample_rows']).round().astype(int))
    exact = []
    with torch.no_grad():
        for start in range(0, len(row_indices), 3):
            color = gt.new_zeros(len(bank.get_xyz), 3)
            subset = row_indices[start:start+3]
            for channel, index in enumerate(subset):
                color[offset+int(index), channel] = 1
            result = renderer(cam, bank, PIPE, gt.new_zeros(3), 0., 0., 0., override_color=color)
            exact.extend(float(result['render'][channel].double().square().sum()) for channel in range(len(subset)))
    exact = np.asarray(exact)
    estimated = np.stack([value[offset:][row_indices].cpu().numpy() for value in estimates])
    active = exact > 1e-6
    rel = np.abs(estimated[:, active]-exact[active])/exact[active]
    exact_checks = {'population_spanning_rows': row_indices.tolist(), 'active_exact_rows': int(active.sum()),
        'median_relative_error_per_seed': np.median(rel, axis=1).tolist() if active.any() else None,
        'p90_relative_error_per_seed': np.quantile(rel, .9, axis=1).tolist() if active.any() else None}
    exact_checks['passed'] = (active.sum() >= plan['exact_sample_active_min']
        and bool((np.median(rel, axis=1) <= plan['exact_relative_median_max']).all())
        and bool((np.quantile(rel, .9, axis=1) <= plan['exact_relative_p90_max']).all()))
    record = {'frame': int(Path(frame['file_path']).stem), 'width': width,
        'independent_seed_agreement': agreement, 'exact_onehot_checks': exact_checks,
        'native_flags_off_RGB_alpha_exact': True,
        'foreground_pixel_count': int(foreground.sum()), 'converged_foreground_fraction': float(good.sum()/foreground.sum()),
        'passed': agreement['passed'] and exact_checks['passed']}
    array = {'support': np.stack([v[offset:].cpu().numpy() for v in estimates]),
        'mass': mass[offset:].cpu().numpy(), 'converged_foreground_mass_fraction': good_fraction[offset:].cpu().numpy(),
        'qualified': qualified.cpu().numpy(), 'exact_sample_rows': row_indices, 'exact_support': exact,
        'positions': bank.get_xyz[offset:].cpu().numpy()}
    with torch.no_grad():
        confidence = torch.zeros_like(qualified)
        order = torch.argsort(torch.where(qualified, (estimates[0][offset:]+estimates[1][offset:])*.5, 0), descending=True, stable=True)
        selected = order[:max(1, int(len(qualified)*plan['top_fraction']))]
        confidence[selected] = qualified[selected]
        colors = gt.new_zeros(len(bank.get_xyz), 3); colors[offset:][confidence] = 1
        anchor_mass = renderer(cam, bank, PIPE, gt.new_zeros(3), 0., 0., 0., override_color=colors)['render']
        sheet = Image.new('RGB', (1200, 250), '#151923'); draw = ImageDraw.Draw(sheet)
        for column, (label, value) in enumerate([('Actual train RGB', gt), ('S3-U unchanged RGB', pred),
                ('Top 2% qualified contribution (not a surface mask)', anchor_mass)]):
            sheet.paste(Image.fromarray(pixels(value)).resize((400, 225)), (column*400, 25))
            draw.text((column*400+6, 5), label, fill='white')
    return record, array, sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-uncertainty-diagnostic', action='store_true')
    parser.add_argument('--protocol', default='observed_pilot_protocol.json')
    args = parser.parse_args()
    if not args.enable_uncertainty_diagnostic:
        raise SystemExit('Observed reliability diagnostic is opt-in')
    path = EXP/args.protocol
    assert path.resolve().parent == EXP.resolve()
    plan = read(path)
    for name, expected in plan['sources'].items():
        assert sha(ROOT/name) == expected, name
    for name, expected in plan['inputs'].items():
        assert sha(ROOT/name) == expected, name
    tiny = read(EXP/'qualification_protocol_v4_native_audit.json')
    assert tiny['passed'] and plan['batches'] in tiny['qualified_batches']
    directory = EXP/plan['output_name']; directory.mkdir(parents=True, exist_ok=True)
    out = OUT/plan['output_name']; out.mkdir(parents=True, exist_ok=True)
    assert not (directory/'result.json').exists(), 'Preserve terminal audit results'
    torch.set_num_threads(4)
    renderer, scene, optimizer, state, original = load_parent()
    before_rng = snapshot_rng(); before_model = model_digest(scene)
    del optimizer, state, original
    meta = read(DATA/'transforms_train.json')
    byid = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    rows = []
    for number, width in plan['cases']:
        start = time.monotonic()
        name = f't{number:04d}_w{width}'
        receipt = directory/(name+'.json')
        if receipt.exists():
            record = read(receipt)
            assert record['protocol_sha256'] == sha(path)
            assert sha(out/(name+'.npz')) == record['array_sha256']
            assert sha(out/(name+'.png')) == record['figure_sha256']
        else:
            record, array, sheet = measure(renderer, scene, byid[number], meta, width, plan)
            np.savez_compressed(out/(name+'.npz'), **array); sheet.save(out/(name+'.png'))
            record.update(protocol_sha256=sha(path), array_sha256=sha(out/(name+'.npz')),
                figure_sha256=sha(out/(name+'.png')), wall_seconds=time.monotonic()-start)
            write(receipt, record)
        rows.append(record)
        print('OBSERVED_RELIABILITY_CASE', json.dumps(record), flush=True)
        assert_rng(before_rng)
    assert model_digest(scene) == before_model
    assert all(p.grad is None for p in scene.parameters())
    assert sha(PARENT) == PARENT_SHA
    for name, expected in plan['sources'].items():
        assert sha(ROOT/name) == expected, name
    write(directory/'result.json', {'passed': all(r['passed'] for r in rows), 'cases': rows,
        'all_model_tensors_and_RNG_unchanged': True, 'model_digest': before_model,
        'optimizer_updates': 0, 'test_RGB_read': False, 'protocol_sha256': sha(path),
        'interpretation': 'Color-support qualification only; no proof of depth, physical identity or full surfaces'})


if __name__ == '__main__':
    main()
