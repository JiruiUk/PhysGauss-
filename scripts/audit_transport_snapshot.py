"""Read-only local distortion audit of a pinned I8 snapshot, never a loss."""
import argparse
from pathlib import Path
import torch

from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.covariance_transport.prepare import EXP
from variants.staged_reconstruction.covariance_transport.core import motion_with_differential, covariance, transport
from variants.staged_reconstruction.invertible_motion.core import InvertibleMotion
from variants.staged_reconstruction.geometry_recovery.model import bounded_scale
from scripts.audit_invertible_strain import quantiles, relative_gradient, summarize


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    path = (ROOT/args.snapshot).resolve()
    assert path.is_relative_to(ROOT/'runs/pidg_staged_reconstruction/covariance_transport')
    assert sha(path) == args.expected_sha256
    output = EXP/f'{args.tag}_strain.json'
    assert not output.exists(), 'Keep diagnostic versions'
    torch.set_num_threads(4)
    state = torch.load(path, map_location='cpu', weights_only=False)
    plan = read(EXP/'protocol.json')
    assert state['protocol_sha256'] == sha(EXP/'protocol.json')
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(918)
        motion = InvertibleMotion(plan['width'], plan['log_scale_bound'], plan['translation_bound'])
    motion.load_state_dict({k.removeprefix('networks.1.'):v for k,v in state['model'].items() if k.startswith('networks.1.')})
    motion = motion.to(device='cuda', dtype=torch.double)
    model = state['model']
    x = model['pools.0._xyz'].to(device='cuda', dtype=torch.double)
    logscale = model['pools.0._scaling'].to(device='cuda', dtype=torch.double)
    q = torch.nn.functional.normalize(model['pools.0._rotation'].to(device='cuda', dtype=torch.double), dim=-1)
    opacity = model['pools.0._opacity'].to(device='cuda', dtype=torch.double).sigmoid().reshape(-1)
    # Canonical scale bounds use the frozen original initialization spacing.
    import numpy as np
    init = ROOT/read(ROOT/'experiments/staged_reconstruction/S3_union/data_receipt.json')['initialization_file']
    with np.load(init) as value:
        spacing = float(value['spacing'])
    scale = bounded_scale(logscale, torch.zeros_like(logscale), spacing, True)
    canonical = covariance(scale, q)
    def differential(t):
        return motion_with_differential(motion, x, x.new_full((len(x), 1), t))[1]
    ref = differential(1/199)
    masks = {'all': torch.ones(len(x), dtype=torch.bool, device='cuda'), 'opacity_above_0.1': opacity>.1}
    rows = []
    for frame in plan['screen_times']:
        j = differential((frame-1)/199)
        relative = relative_gradient(j, ref)
        effective = torch.linalg.eigvalsh(transport(j, canonical))
        assert (effective>0).all() and j.isfinite().all()
        axis = (effective[:, -1]/effective[:, 0]).sqrt()
        row = {'frame':frame, 'spatial_map':{name:summarize(j,mask) for name,mask in masks.items()},
               'relative_to_frame2':{name:summarize(relative,mask) for name,mask in masks.items()},
               'render_covariance_axis_ratio':{name:quantiles(axis[mask]) for name,mask in masks.items()}}
        rows.append(row)
        print('TRANSPORT_SNAPSHOT_STRAIN',state['iteration'],frame,'min_q01',row['spatial_map']['all']['minimum_principal_stretch']['0.01'],
              'max_q99',row['spatial_map']['all']['maximum_principal_stretch']['0.99'],flush=True)
    assert sha(path) == args.expected_sha256
    write(output,{'checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':args.expected_sha256,
                  'iteration':state['iteration'],'script_sha256':sha(Path(__file__)),'dtype':'float64','rows':rows,
                  'optimizer_updates':0,'RGB_read':False,'not_constitutive_F_or_independent_material_truth':True,
                  'scope':'Fixed Gaussian centers and nine screen times. Frame2 is not known stress-free. Descriptive distortion statistics, not added gates, causality or final completeness.'})


if __name__ == '__main__':
    main()
