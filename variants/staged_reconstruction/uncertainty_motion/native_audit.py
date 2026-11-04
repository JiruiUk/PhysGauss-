"""Opt-in native color-support qualification, never an optimizer update."""
import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.stats import spearmanr
from ..prepare_data import ROOT, DATA, read, write, sha
from ..train_static import setup, PIPE
from .core import color_vjp, squared_support

EXP = ROOT/'experiments/staged_reconstruction/uncertainty_motion'
OUT = ROOT/'output/staged_reconstruction/uncertainty_motion'
NAMES = ['get_xyz', 'get_scaling', 'get_rotation', 'get_opacity', 'get_features']


def detached_bank(bank):
    return SimpleNamespace(max_sh_degree=3, active_sh_degree=3,
        **{key: getattr(bank, key).detach().clone() for key in NAMES})


def snapshot_rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            [v.clone() for v in torch.cuda.get_rng_state_all()])


def assert_rng(before):
    after = snapshot_rng()
    assert before[0] == after[0]
    assert all(np.array_equal(a, b) for a, b in zip(before[1], after[1]))
    assert torch.equal(before[2], after[2])
    assert all(torch.equal(a, b) for a, b in zip(before[3], after[3]))


def support_render(renderer, cam, bank):
    colors = torch.zeros(len(bank.get_xyz), 3, device='cuda', requires_grad=True)
    package = renderer(cam, bank, PIPE, colors.new_zeros(3), 0., 0., 0., override_color=colors)
    return package, colors


def relative_metrics(estimate, exact):
    good = exact > 1e-6
    a, b = estimate[good].cpu().numpy(), exact[good].cpu().numpy()
    error = np.abs(a-b)/b
    k = max(1, len(a)//5)
    overlap = len(set(np.argsort(a)[-k:]) & set(np.argsort(b)[-k:]))/k
    return {'active_points': len(a), 'relative_error_median': float(np.median(error)),
        'relative_error_p90': float(np.quantile(error, .9)),
        'spearman': float(spearmanr(a, b).statistic), 'top20_overlap': overlap,
        'max_abs_error': float((estimate-exact).abs().max())}


def tiny(protocol):
    from utils.graphics_utils import getProjectionMatrix
    renderer = setup()
    count = 37
    xyz = torch.zeros(count, 3, device='cuda')
    xyz[:32, 0] = .26; xyz[:32, 2] = 2.+.014*torch.arange(32, device='cuda')
    # Foreground occluder and a rear point with matching image coordinates.
    xyz[32] = xyz.new_tensor([-.28, 0., 1.8]); xyz[33] = xyz[32]*1.35
    xyz[34] = xyz.new_tensor([-.3, .25, 2.]); xyz[35] = xyz.new_tensor([.04, -.22, 2.1])
    xyz[36] = xyz.new_tensor([20., 0., 2.])  # off-screen point must have zero support
    scales = torch.full((count, 3), .075, device='cuda')
    scales[32] = .13
    opacity = torch.full((count, 1), .065, device='cuda'); opacity[32] = .95
    q = torch.zeros(count, 4, device='cuda'); q[:, 0] = 1
    bank = SimpleNamespace(get_xyz=xyz, get_scaling=scales, get_rotation=q, get_opacity=opacity,
        get_features=torch.zeros(count, 16, 3, device='cuda'), max_sh_degree=3, active_sh_degree=3)
    projection = getProjectionMatrix(znear=.01, zfar=100., fovX=1., fovY=.6).T.cuda()
    cam = SimpleNamespace(FoVx=1., FoVy=.6, image_width=96, image_height=54,
        world_view_transform=torch.eye(4, device='cuda'), full_proj_transform=projection,
        camera_center=torch.zeros(3, device='cuda'))
    before = snapshot_rng(); saved = {n: getattr(bank, n).clone() for n in NAMES}
    exact_rows = []
    with torch.no_grad():
        for start in range(0, count, 3):
            color = torch.zeros(count, 3, device='cuda')
            n = min(3, count-start)
            for k in range(n):
                color[start+k, k] = 1
            package = renderer(cam, bank, PIPE, color.new_zeros(3), 0., 0., 0., override_color=color)
            exact_rows.extend(package['render'][k].double().flatten() for k in range(n))
    a = torch.stack(exact_rows)
    exact = a.square().sum(1)
    package, colors = support_render(renderer, cam, bank)
    linear = color_vjp(package['render'], colors, torch.ones_like(package['render'][0]))
    linear_error = float((linear.double()-a.sum(1)).abs().max())
    assert torch.allclose(linear.double(), a.sum(1), rtol=1e-4, atol=1e-5), linear_error
    per_seed = []
    for batches in protocol.get('batch_schedule', [protocol['batches']]):
        for seed in protocol['seeds']:
            result = squared_support(package['render'], colors, batches=batches, seed=seed)
            metric = relative_metrics(result['support'], exact)
            metric.update(seed=seed, batches=batches, samples=result['samples'])
            metric['passed'] = (metric['relative_error_median'] <= protocol['tiny_median_relative_error_max']
                and metric['relative_error_p90'] <= protocol['tiny_p90_relative_error_max']
                and metric['spearman'] >= protocol['tiny_spearman_min']
                and metric['top20_overlap'] >= protocol['tiny_top20_overlap_min'])
            per_seed.append(metric)
            print('TINY_SUPPORT_SAMPLE_CONVERGENCE', json.dumps(metric), flush=True)
            assert result['support'][-1] == 0 and exact[-1] == 0
    ids = package['gs_per_pixel'].reshape(-1)
    weight = package['weight_per_gs_pixel'].detach().double().reshape(-1)
    exported = torch.zeros(count, dtype=torch.float64, device='cuda')
    valid = (ids >= 0) & (ids < count)
    assert bool((ids[~valid] == -1).all()) and bool((weight[~valid] == 0).all())
    exported.scatter_add_(0, ids[valid], weight[valid].square())
    export_difference = float((exported-exact).abs().sum()/exact.sum())
    assert export_difference > .01
    assert bool((a[20:32].sum(1) > 0).all()), 'Explicitly verify contributions beyond row20'
    assert_rng(before)
    assert all(torch.equal(getattr(bank, n), saved[n]) for n in NAMES) and colors.grad is None
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/'tiny_exact_weights.npz', exact_color_weights=a.cpu().numpy(),
        squared_support=exact.cpu().numpy(), truncated_export_support=exported.cpu().numpy())
    passing = [b for b in protocol.get('batch_schedule', [protocol['batches']])
               if all(v['passed'] for v in per_seed if v['batches'] == b)]
    return {'passed': bool(passing), 'qualified_batches': passing, 'native_renderer_width': 96, 'points': count,
        'exact_onehot_rgb_renders': (count+2)//3, 'exact_linear_vjp_max_abs_error': linear_error,
        'independent_seeds': per_seed, 'export_buffer_relative_L1_error': export_difference,
        'later_than20_contributions_explicitly_observed': True,
        'all_global_RNG_and_geometry_unchanged': True, 'optimizer_updates': 0,
        'explicit_weights_sha256': sha(OUT/'tiny_exact_weights.npz')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-uncertainty-diagnostic', action='store_true')
    parser.add_argument('--protocol', default='qualification_protocol.json')
    args = parser.parse_args()
    if not args.enable_uncertainty_diagnostic:
        raise SystemExit('Reliability audit is opt-in')
    torch.set_num_threads(4)
    protocol_path = EXP/args.protocol
    assert protocol_path.resolve().parent == EXP.resolve()
    protocol = read(protocol_path)
    for name, expected in protocol['sources'].items():
        assert sha(ROOT/name) == expected, name
    receipt = EXP/(protocol_path.stem+'_native_audit.json')
    assert not receipt.exists(), 'Preserve diagnostic receipts'
    # Setup the unchanged renderer before importing its camera helper.
    setup()
    result = tiny(protocol)
    result['qualification_protocol_sha256'] = sha(protocol_path)
    write(receipt, result)
    print('UNCERTAINTY_TINY_NATIVE_RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
