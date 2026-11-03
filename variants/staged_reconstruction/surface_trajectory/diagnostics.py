"""Read-only future support audit after bootstrap admission; never filters bindings."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from ..prepare_data import ROOT, REPORT, DATA, read, write, sha
from ..train_static import setup, camera, PIPE
from ..stage2.model import compose
from ..stage2.train import image
from ..union.prepare import verify_sources
from .prepare import OUT, PARENT, PARENT_SHA
from .observations import appearance_masks


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-surface-trajectory', action='store_true')
    args = parser.parse_args()
    if not args.enable_surface_trajectory:
        raise SystemExit('Surface diagnostics is opt-in')
    out = OUT/'landmark_v3'
    path = out/'future_support.json'
    assert not path.exists(), 'Preserve completed support audit'
    bindings = read(out/'bindings.json')['bindings']
    protocol = read(out/'binding_protocol.json')
    assert sha(PARENT) == PARENT_SHA
    verify_sources()
    torch.set_num_threads(4)
    renderer = setup()
    from ..union.model import UnionScene
    scene = UnionScene(read(REPORT/'S3_union/protocol.json'))
    state = torch.load(PARENT, map_location='cpu', weights_only=False)
    scene.load_state_dict(state['model'])
    scene.requires_grad_(False)
    meta = read(DATA/'transforms_train.json')
    frames = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    queries = defaultdict(list)
    for item in bindings:
        for index, (number, uv) in enumerate(zip(item['frames'], item['uv_400'])):
            queries[number].append((item, index, uv))
    observations = []
    for number, requests in sorted(queries.items()):
        frame = frames[number]
        cam = camera(frame, meta, 400)
        g = scene.dynamic_at(frame['time'])
        full = compose(scene.static, [g])
        offset = len(scene.static.get_xyz)
        xyz = g.get_xyz
        mask = torch.tensor(appearance_masks(image(frame, 400), 2, False)[0], device='cuda')
        yy, xx = torch.meshgrid(torch.arange(225, device='cuda'), torch.arange(400, device='cuda'), indexing='ij')
        color = torch.zeros((len(full.get_xyz), 3), device='cuda')
        for start in range(0, len(requests), 3):
            batch = requests[start:start+3]
            color.zero_()
            for channel, (item, _, _) in enumerate(batch):
                active = [r for r, w in zip(item['rows'], item['weights']) if w >= 1e-4]
                color[torch.tensor(active, device='cuda')+offset, channel] = 1.
            contribution = renderer(cam, full, PIPE, xyz.new_zeros(3), 0., 0., 0., override_color=color)['render'][:len(batch)]
            for channel, (item, index, target) in enumerate(batch):
                rows = torch.tensor(item['rows'], device='cuda')
                weights = xyz.new_tensor(item['weights'])
                center = (xyz[rows]*weights[:, None]).sum(0)
                h = torch.cat([center, center.new_ones(1)])@cam.full_proj_transform
                valid = bool(torch.isfinite(h).all() and h[3] > 0)
                if valid:
                    uv = ((h[:2]/h[3]+1)*h.new_tensor([400, 225])-1)/2
                    error = float((uv-uv.new_tensor(target)).norm())
                    predicted = uv.cpu().tolist()
                else:
                    error, predicted = None, None
                patch = (xx-target[0]).square()+(yy-target[1]).square() <= protocol['patch_radius_px']**2
                total = float(contribution[channel].sum())
                foreground = float(contribution[channel][mask].sum())
                patch_mass = float(contribution[channel][patch].sum())
                supported = (valid and total >= protocol['minimum_total_mass']
                             and foreground >= protocol['foreground_purity']*total
                             and patch_mass >= protocol['minimum_patch_mass'])
                observations.append(dict(track_id=item['track_id'], frame=number, bootstrap=index < 3,
                    time_band=item['time_band'], predicted_uv_400=predicted, target_uv_400=target,
                    projection_valid=valid, error_px=error, support_total_mass=total,
                    support_foreground_mass=foreground, support_patch_mass=patch_mass,
                    image_support_qualified=supported,
                    support_spread=float(((xyz[rows]-center).square().sum(1)*weights).sum().sqrt()),
                    weighted_opacity=float((g.get_opacity[rows].flatten()*weights).sum())))
    summaries = {}
    for name, subset in [('bootstrap', [o for o in observations if o['bootstrap']]),
                         ('future', [o for o in observations if not o['bootstrap']])]:
        values = [o['error_px'] for o in subset if o['error_px'] is not None]
        summaries[name] = dict(observations=len(subset), valid_projections=len(values),
            mean_error_px=float(np.mean(values)) if values else None,
            median_error_px=float(np.median(values)) if values else None,
            p90_error_px=float(np.quantile(values, .9)) if values else None,
            image_supported=sum(o['image_support_qualified'] for o in subset),
            note='Fixed qualification; invalid/lost support is reported, never dropped from the denominator')
    assert all(torch.equal(v.cpu(), state['model'][k]) for k, v in scene.state_dict().items())
    assert sha(PARENT) == PARENT_SHA and all(p.grad is None for p in scene.parameters())
    verify_sources()
    write(path, dict(parent_sha256=PARENT_SHA, binding_sha256=sha(out/'bindings.json'),
        summaries=summaries, observations=observations, optimizer_updates=0,
        admission_used_future_errors=False, model_tensors_unchanged=True,
        material_identity_GT=False, not_a_new_model_or_training_result=True))
    print('FUTURE_SUPPORT', summaries, flush=True)


if __name__ == '__main__':
    main()
