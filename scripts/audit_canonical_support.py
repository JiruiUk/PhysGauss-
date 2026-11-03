"""Read-only train-view support/gradient audit of an explicitly pinned model.

No adaptive density operation is performed. Visibility here is the rasterizer's
radius/frustum admission, not known physical visibility or a correspondence.
"""
import argparse
from pathlib import Path
import re
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
import torch

from variants.staged_reconstruction.prepare_data import ROOT, REPORT, RUNS, DATA, read, write, sha
from variants.staged_reconstruction.early_depth.train import load_initial
from variants.staged_reconstruction.stage2.train import image
from variants.staged_reconstruction.train_static import camera, PIPE
from variants.staged_reconstruction.union.core import dynamic_mask
from variants.staged_reconstruction.union.train import reconstruction
from variants.staged_reconstruction.covariance_transport.runtime import install, renderer_for


def quantiles(values):
    values = np.asarray(values)
    if not len(values):
        return {}
    return {str(q): float(np.quantile(values, q)) for q in [0., .01, .1, .5, .9, .99, 1.]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['S3U', 'I8', 'I9'], required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    assert re.fullmatch(r'[a-z0-9_-]+', args.tag)
    path = (ROOT/args.snapshot).resolve()
    branch = {'S3U': 'S3_union', 'I8': 'covariance_transport', 'I9': 'relative_strain'}[args.arm]
    assert path.is_relative_to((RUNS/branch).resolve())
    assert sha(path) == args.expected_sha256
    output = REPORT/'canonical_support'/f'{args.tag}.json'
    assert not output.exists(), 'Preserve diagnostic versions'
    torch.set_num_threads(4)
    renderer, scene, opt, initial, _ = load_initial()
    del opt, initial
    if args.arm != 'S3U':
        plan = read(REPORT/branch/'protocol.json')
        install(scene, plan, True)
        renderer = renderer_for(renderer)
    state = torch.load(path, map_location='cpu', weights_only=False)
    if args.arm != 'S3U':
        assert state['protocol_sha256'] == sha(REPORT/branch/'protocol.json')
    iteration = state['iteration']
    scene.load_state_dict(state['model']); del state
    before = {name: value.detach().clone() for name, value in scene.state_dict().items()}
    meta = read(DATA/'transforms_train.json')
    assert len(meta['frames']) == 120
    count = len(scene.pools[0].get_xyz); offset = len(scene.static.get_xyz)
    totals = np.zeros(count, dtype=np.float64)
    admitted = np.zeros(count, dtype=np.int64)
    nonzero = np.zeros(count, dtype=np.int64)
    maximum_radius = np.zeros(count, dtype=np.float64)
    records = []
    for index, frame in enumerate(meta['frames']):
        rgba = image(frame, 800); cam = camera(frame, meta, 800)
        gt = torch.tensor(rgba[..., :3]*rgba[..., 3:4], device='cuda').permute(2, 0, 1)
        scene.zero_grad(set_to_none=True)
        package = scene.render(renderer, cam, frame['time'])
        loss = reconstruction(package, gt, rgba, scene, cam)
        loss.backward()
        gradient = package['viewspace_points'].grad
        assert gradient is not None and gradient.isfinite().all()
        assert len(gradient) == offset+count
        gradient = gradient[offset:, :2].norm(dim=-1).detach().cpu().numpy()
        visible = package['visibility_filter'][offset:].detach().cpu().numpy().astype(bool)
        radius = package['radii'][offset:].detach().cpu().numpy()
        assert np.isfinite(gradient).all() and np.isfinite(radius).all()
        totals += np.where(visible, gradient, 0.)
        admitted += visible; nonzero += visible & (gradient > 1e-12)
        maximum_radius = np.maximum(maximum_radius, radius)
        with torch.no_grad():
            dyn = renderer(cam, scene.dynamic_at(frame['time']), PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
            alpha = dyn['alpha'].squeeze().clamp(0, 1)
            mask = torch.tensor(binary_erosion(dynamic_mask(rgba), iterations=2), device='cuda')
            assert mask.any() and alpha.isfinite().all()
            values = alpha[mask]
            record = {'frame': int(Path(frame['file_path']).stem), 'time': float(frame['time']),
                      'image_objective': float(loss.detach()),
                      'eroded_foreground_pixels': int(mask.sum()),
                      'dynamic_only_alpha_mean_on_eroded_foreground': float(values.mean()),
                      'fraction_eroded_foreground_alpha_below_0.5': float((values < .5).float().mean()),
                      'fraction_eroded_foreground_alpha_below_0.9': float((values < .9).float().mean()),
                      'radius_admitted_rows': int(visible.sum()),
                      'rows_with_nonzero_viewspace_gradient': int((visible & (gradient > 1e-12)).sum())}
        records.append(record)
        del package, dyn, loss, gt, gradient
        if (index+1) % 20 == 0:
            print('CANONICAL_SUPPORT_AUDIT', args.arm, iteration, index+1, '/', len(meta['frames']), flush=True)
    average = totals/np.maximum(admitted, 1)
    pool = scene.pools[0]
    xyz = pool.get_xyz.detach().cpu().numpy()
    opacity = pool.get_opacity.detach().cpu().numpy().reshape(-1)
    active = opacity > .1
    nearest = cKDTree(xyz).query(xyz, k=2, workers=1)[0][:, 1]
    shift = (pool.get_xyz-pool.initial_xyz).detach().abs().amax(-1).cpu().numpy()
    arrays = ROOT/'output/staged_reconstruction/canonical_support'/f'{args.tag}_per_point.npz'
    arrays.parent.mkdir(parents=True, exist_ok=True)
    assert not arrays.exists()
    np.savez_compressed(arrays, admitted_count=admitted, nonzero_gradient_count=nonzero,
                        mean_admitted_viewspace_gradient=average, max_radius_px=maximum_radius,
                        opacity=opacity, canonical_nearest_distance=nearest, canonical_max_shift=shift)
    aggregates = {}
    for name, mask in {'all': np.ones(count, dtype=bool), 'opacity_above_0.1': active}.items():
        aggregates[name] = {'count': int(mask.sum()),
            'fraction_never_radius_admitted': float((admitted[mask] == 0).mean()),
            'fraction_never_nonzero_image_viewspace_gradient': float((nonzero[mask] == 0).mean()),
            'mean_viewspace_gradient_quantiles': quantiles(average[mask]),
            'fraction_above_original_3dgs_threshold_0.0002_descriptive_only': float((average[mask] >= .0002).mean()),
            'max_radius_pixels_quantiles': quantiles(maximum_radius[mask]),
            'canonical_nearest_distance_quantiles': quantiles(nearest[mask]),
            'fraction_near_original_canonical_coordinate_clamp': float((shift[mask] >= .3999).mean())}
    assert all(torch.equal(value, scene.state_dict()[key]) for key, value in before.items())
    assert sha(path) == args.expected_sha256
    write(output, {'arm': args.arm, 'iteration': iteration, 'checkpoint': path.relative_to(ROOT).as_posix(),
                   'checkpoint_sha256': args.expected_sha256, 'script_sha256': sha(Path(__file__)),
                   'training_metadata_sha256': sha(DATA/'transforms_train.json'),
                   'native_width': 800, 'training_frames': 120, 'dynamic_rows': count, 'static_rows': offset,
                   'optimizer_updates': 0, 'all_model_tensors_unchanged': True, 'test_RGB_read': False,
                   'new_view_GT': False, 'density_operations': 0,
                   'per_point_arrays': arrays.relative_to(ROOT).as_posix(), 'per_point_arrays_sha256': sha(arrays),
                   'aggregates': aggregates, 'training_frame_records': records,
                   'scope': 'Observed train-view footprint/gradient diagnosis only. Radius admission is not physical visibility. Original3DGS threshold is not calibrated for this loss/resolution. No causal density ablation, material/object labels, topology claim, hidden-surface GT, completeness acceptance or training modification.'})
    print('CANONICAL_SUPPORT_AUDIT_SAVED', output, flush=True)


if __name__ == '__main__': main()
