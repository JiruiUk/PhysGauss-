"""Read-only I7 spatial/relative stretch diagnostic; no RGB, renderer or optimizer."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/staged_reconstruction/invertible_motion'
CHECKPOINT = ROOT/'runs/pidg_staged_reconstruction/invertible_motion/formal/invertible/checkpoint.pt'
EXPECTED_SHA = 'e5eb8b708051a757112a60a9e58f27cea616ab290a4e1d589012c94f15c05190'


def pointwise_jacobian(mapping, points):
    """Exact Nx3x3 derivative ONLY for maps acting independently on each row."""
    with torch.enable_grad():
        query = points.detach().clone().requires_grad_(True)
        mapped = mapping(query)
        if mapped.shape != query.shape or query.ndim != 2 or query.shape[-1] != 3:
            raise ValueError('Expected independent Nx3 input/output')
        rows = [torch.autograd.grad(mapped[:, k].sum(), query, retain_graph=k < 2)[0]
                for k in range(3)]
    return torch.stack(rows, 1).detach()


def relative_gradient(current, reference):
    """D phi_t(X) D phi_ref(X)^-1; reference is model-conditioned, not relaxed."""
    return torch.linalg.solve(reference.transpose(-1, -2), current.transpose(-1, -2)).transpose(-1, -2)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def quantiles(value):
    return {str(q): float(torch.quantile(value, q)) for q in [0., .01, .1, .5, .9, .99, 1.]}


def summarize(jacobian, mask):
    value = jacobian[mask]
    if len(value) == 0:
        return {'count': 0}
    s = torch.linalg.svdvals(value)
    det = torch.linalg.det(value)
    return {'count': len(value), 'minimum_principal_stretch': quantiles(s[:, -1]),
            'maximum_principal_stretch': quantiles(s[:, 0]),
            'condition_number': quantiles(s[:, 0]/s[:, -1]), 'determinant': quantiles(det),
            'fraction_min_below_quarter': float((s[:, -1] < .25).double().mean()),
            'fraction_max_above_four': float((s[:, 0] > 4).double().mean()),
            'fraction_condition_above_sixteen': float((s[:, 0]/s[:, -1] > 16).double().mean()),
            'fraction_negative_determinant': float((det < 0).double().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-read-only-diagnostic', action='store_true')
    args = parser.parse_args()
    if not args.run_read_only_diagnostic:
        raise SystemExit('Actual checkpoint diagnostic is opt-in')
    output = EXP/'strain_diagnostic.json'
    raw_path = ROOT/'runs/pidg_staged_reconstruction/invertible_motion/strain_diagnostic.npz'
    assert not output.exists() and not raw_path.exists(), 'Preserve diagnostic versions'
    torch.set_num_threads(4)
    from variants.staged_reconstruction.invertible_motion.core import InvertibleMotion
    from variants.staged_reconstruction.invertible_motion.freeze import verify
    verify()
    assert digest(CHECKPOINT) == EXPECTED_SHA
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert state['iteration'] == 16000
    model_state = state['model']
    plan = json.loads((EXP/'protocol.json').read_text())
    # Independent local copy: no scene, optimizer, RGB or covariance fields loaded.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(918)
        motion = InvertibleMotion(plan['width'], plan['log_scale_bound'], plan['translation_bound'])
    motion.load_state_dict({key.removeprefix('networks.1.'): value for key, value in model_state.items()
                            if key.startswith('networks.1.')}, strict=True)
    motion = motion.to(device='cuda', dtype=torch.double).eval().requires_grad_(False)
    points = model_state['pools.0._xyz'].to(device='cuda', dtype=torch.double)
    opacity = model_state['pools.0._opacity'].sigmoid().reshape(-1).to(device='cuda', dtype=torch.double)
    del state, model_state
    def jacobians(time):
        values = []
        for chunk in points.split(512):
            times = chunk.new_full((len(chunk), 1), time)
            values.append(pointwise_jacobian(lambda query: motion(query, times), chunk))
        return torch.cat(values)
    reference = jacobians((2-1)/199)
    masks = {'all': torch.ones(len(points), dtype=torch.bool, device=points.device),
             'canonical_opacity_above_0.1': opacity > .1}
    rows, raw = [], {'canonical_xyz': points.cpu().numpy(), 'canonical_opacity': opacity.cpu().numpy()}
    for frame in plan['screen_times']:
        t = (frame-1)/199
        j = reference if frame == 2 else jacobians(t)
        relative = relative_gradient(j, reference)
        assert j.isfinite().all() and relative.isfinite().all()
        with torch.no_grad():
            _, logdet = motion(points, points.new_full((len(points), 1), t), return_logdet=True)
        logdet_error = float((torch.linalg.det(j).log()-logdet).abs().max())
        assert logdet_error < 1e-8
        # Direct central differences on fixed, evenly spaced canonical rows.
        sample = points[torch.linspace(0, len(points)-1, 32, device=points.device).long()]
        sample_t = sample.new_full((len(sample), 1), t)
        eps = 1e-5
        with torch.no_grad():
            finite = torch.stack([(motion(sample+torch.eye(3, device=points.device, dtype=points.dtype)[k]*eps, sample_t)-
                                   motion(sample-torch.eye(3, device=points.device, dtype=points.dtype)[k]*eps, sample_t))/(2*eps)
                                  for k in range(3)], -1)
        exact = pointwise_jacobian(lambda query: motion(query, sample_t), sample)
        finite_error = float((finite-exact).abs().max())
        assert finite_error < 1e-5
        row = {'frame': frame, 'spatial_map': {name: summarize(j, mask) for name, mask in masks.items()},
               'relative_to_frame2': {name: summarize(relative, mask) for name, mask in masks.items()},
               'analytic_logdet_max_abs_difference': logdet_error,
               'finite_difference_max_abs_difference_32_fixed_rows': finite_error}
        rows.append(row)
        raw[f'jacobian_{frame:04d}'] = j.cpu().numpy()
        raw[f'relative_{frame:04d}'] = relative.cpu().numpy()
        print('INVERTIBLE_STRAIN', frame, 'map', row['spatial_map']['all']['fraction_min_below_quarter'],
              'relative', row['relative_to_frame2']['all']['fraction_min_below_quarter'], flush=True)
    assert digest(CHECKPOINT) == EXPECTED_SHA
    np.savez_compressed(raw_path, **raw)
    record = {'checkpoint': CHECKPOINT.relative_to(ROOT).as_posix(), 'checkpoint_sha256': EXPECTED_SHA,
              'checkpoint_iteration': 16000, 'dtype': 'float64', 'reference_frame': 2, 'rows': rows,
              'raw_arrays': raw_path.relative_to(ROOT).as_posix(), 'raw_arrays_sha256': digest(raw_path),
              'script_sha256': digest(Path(__file__)), 'tests_sha256': digest(ROOT/'tests/test_invertible_strain_diagnostic.py'),
              'optimizer_updates': 0, 'RGB_read': False, 'checkpoint_unchanged': True,
              'constitutive_branch_or_GT_material_F': False,
              'scope': 'Exact local derivative of independent smooth I7 position map. Relative derivatives follow the same canonical model row; frame2 is not a known stress-free reference. Canonical opacity is not visibility or rendered contribution. Thresholds are descriptive diagnostics, not admission gates. No independent material correspondence or causal proof.'}
    output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print('INVERTIBLE_STRAIN_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
