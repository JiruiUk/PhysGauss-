"""Read-only audit of the actual rasterizer's raw quaternion covariance path."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from scripts.audit_invertible_strain import ROOT, EXP, CHECKPOINT, EXPECTED_SHA, digest, quantiles


def raster_covariance(scales, quaternion):
    """Match released CUDA computeCov3D, including its disabled normalization."""
    r, x, y, z = quaternion.unbind(-1)
    rotation = torch.stack([1-2*(y*y+z*z), 2*(x*y-r*z), 2*(x*z+r*y),
                            2*(x*y+r*z), 1-2*(x*x+z*z), 2*(y*z-r*x),
                            2*(x*z-r*y), 2*(y*z+r*x), 1-2*(x*x+y*y)], -1).reshape(-1, 3, 3)
    factor = rotation*scales[:, None, :]
    return factor@factor.transpose(-1, -2)


def covariance_axis_ratio(current, reference):
    """Generalized eigen axis ratios; both matrices must be positive definite."""
    lower = torch.linalg.cholesky(reference)
    left = torch.linalg.solve_triangular(lower, current, upper=False)
    white = torch.linalg.solve_triangular(lower, left.transpose(-1, -2), upper=False).transpose(-1, -2)
    eigenvalues = torch.linalg.eigvalsh((white+white.transpose(-1, -2))/2)
    if not (eigenvalues > 0).all():
        raise ValueError('Covariance ratio requires strictly positive eigenvalues')
    return eigenvalues.sqrt()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-read-only-diagnostic', action='store_true')
    args = parser.parse_args()
    if not args.run_read_only_diagnostic:
        raise SystemExit('Actual covariance diagnostic is opt-in')
    output = EXP/'covariance_diagnostic.json'
    assert not output.exists()
    torch.set_num_threads(4)
    from variants.staged_reconstruction.early_depth.train import load_initial
    from variants.staged_reconstruction.invertible_motion.core import install
    from variants.staged_reconstruction.invertible_motion.freeze import verify
    from variants.staged_reconstruction.train_static import camera, PIPE
    from variants.staged_reconstruction.stage2.model import compose
    from variants.staged_reconstruction.prepare_data import DATA, read
    from types import SimpleNamespace
    verify()
    assert digest(CHECKPOINT) == EXPECTED_SHA
    renderer, scene, opt, initial, _ = load_initial()
    del opt, initial
    plan = read(EXP/'protocol.json')
    install(scene, plan, True)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    scene.load_state_dict(state['model']); del state
    raw_path = ROOT/read(EXP/'strain_diagnostic.json')['raw_arrays']
    assert digest(raw_path) == read(EXP/'strain_diagnostic.json')['raw_arrays_sha256']
    raw = np.load(raw_path)
    ref = scene.dynamic_at(1/199)
    reference = raster_covariance(ref.get_scaling.double(), ref.get_rotation.double())
    masks = {'all': torch.ones(len(ref.get_xyz), dtype=torch.bool, device='cuda'),
             'canonical_opacity_above_0.1': ref.get_opacity.reshape(-1) > .1}
    rows = []
    for frame in plan['screen_times']:
        current = scene.dynamic_at((frame-1)/199)
        q, s = current.get_rotation.double(), current.get_scaling.double()
        c = raster_covariance(s, q)
        normalized = raster_covariance(s, torch.nn.functional.normalize(q, dim=-1))
        f = torch.tensor(raw[f'relative_{frame:04d}'], device='cuda')
        transported = f@reference@f.transpose(-1, -2)
        ratios = covariance_axis_ratio(c, transported)
        norm_ratios = covariance_axis_ratio(c, normalized)
        strata = {}
        for name, mask in masks.items():
            r, n = ratios[mask], norm_ratios[mask]
            strata[name] = {'count': int(mask.sum()), 'raw_quaternion_norm': quantiles(q[mask].norm(dim=-1)),
                            'scale_axis_ratio': quantiles(s[mask].max(-1).values/s[mask].min(-1).values),
                            'actual_covariance_axis_ratio': quantiles(torch.linalg.eigvalsh(c[mask]).sqrt()[:, -1]/torch.linalg.eigvalsh(c[mask]).sqrt()[:, 0]),
                            'relative_transport_min_axis_ratio': quantiles(r[:, 0]),
                            'relative_transport_max_axis_ratio': quantiles(r[:, -1]),
                            'fraction_relative_axis_below_quarter_or_above_four': float(((r[:, 0]<.25)|(r[:, -1]>4)).double().mean()),
                            'raw_vs_normalized_min_axis_ratio': quantiles(n[:, 0]),
                            'raw_vs_normalized_max_axis_ratio': quantiles(n[:, -1]),
                            'fraction_raw_vs_normalized_axis_below_quarter_or_above_four': float(((n[:, 0]<.25)|(n[:, -1]>4)).double().mean())}
        rows.append({'frame': frame, 'strata': strata})
        print('RENDER_COVARIANCE', frame, 'q99', strata['all']['raw_quaternion_norm']['0.99'],
              'transport_outliers', strata['all']['fraction_relative_axis_below_quarter_or_above_four'], flush=True)
    # Native rasterizer parity establishes that the Python audit uses actual CUDA conventions.
    meta = read(DATA/'transforms_train.json'); parity = []
    for number in [43, 146]:
        frame = next(frame for frame in meta['frames'] if int(Path(frame['file_path']).stem) == number)
        cam = camera(frame, meta, 1600)
        bank = compose(scene.static, [scene.dynamic_at(frame['time'])])
        expected = renderer(cam, bank, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
        covariance = raster_covariance(bank.get_scaling, bank.get_rotation)
        packed = covariance[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]].contiguous()
        bank.get_covariance = lambda modifier: packed*modifier**2
        alternate = renderer(cam, bank, SimpleNamespace(compute_cov3D_python=True, convert_SHs_python=False, debug=False), scene.centers.new_zeros(3), 0., 0., 0.)
        rgb_error = float((expected['render']-alternate['render']).abs().max())
        alpha_error = float((expected['alpha']-alternate['alpha']).abs().max())
        print('COVARIANCE_PARITY', number, rgb_error, alpha_error,
              'RGB_mean', float((expected['render']-alternate['render']).abs().mean()),
              'conic', float((expected['conic_2D']-alternate['conic_2D']).abs().max()), flush=True)
        parity.append({'frame': number, 'native_width': 1600, 'rgb_max_abs': rgb_error, 'alpha_max_abs': alpha_error,
                       'rgb_mean_abs': float((expected['render']-alternate['render']).abs().mean()),
                       'alpha_mean_abs': float((expected['alpha']-alternate['alpha']).abs().mean()),
                       'conic_max_abs': float((expected['conic_2D']-alternate['conic_2D']).abs().max()),
                       'original_max_pixel_tolerance_3e4_passed': rgb_error < 3e-4 and alpha_error < 3e-4})
    raw.close()
    assert digest(CHECKPOINT) == EXPECTED_SHA
    record = {'checkpoint_sha256': EXPECTED_SHA, 'rows': rows, 'native_covariance_parity': parity,
              'raw_quaternion_normalization_disabled_in_original_CUDA': True,
              'CUDA_source': 'third_party/PIDG/submodules/flow-based-diff-gaussian-rasterization/cuda_rasterizer/forward.cu',
              'CUDA_source_sha256': digest(ROOT/'third_party/PIDG/submodules/flow-based-diff-gaussian-rasterization/cuda_rasterizer/forward.cu'),
              'script_sha256': digest(Path(__file__)), 'tests_sha256': digest(ROOT/'tests/test_render_covariance_diagnostic.py'),
              'optimizer_updates': 0, 'RGB_read': False, 'checkpoint_unchanged': True,
              'native_pixel_parity_passed': all(row['original_max_pixel_tolerance_3e4_passed'] for row in parity),
              'parity_limitations': 'Original maximum-pixel tolerance failed at frame146 in both earlier attempts. Preserve those logs and report maximum and mean pixel differences, plus projected conics; do not claim exact renderer equivalence or silently weaken the tolerance. This script is diagnostic only, not an admitted alternate training renderer.',
              'scope': 'Conditional covariance consistency diagnostic. Gaussian covariance is rendering support, not a proven physical material volume. Positive-definite ratio and threshold statistics are not physical GT or causality. Native parity renders use metadata poses only.'}
    output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print('RENDER_COVARIANCE_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
