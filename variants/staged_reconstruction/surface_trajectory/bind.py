"""R1 immutable S3-U bootstrap binding, exact forward contribution checks."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from ..prepare_data import ROOT, REPORT, RUNS, DATA, read, write, sha
from ..train_static import setup, camera, PIPE
from ..stage2.model import compose
from ..stage2.train import image
from ..union.core import dynamic_mask
from ..union.prepare import verify_sources
from .prepare import OUT, PARENT, PARENT_SHA
from .binding import choose_support
from .observations import appearance_masks

CONFIG = dict(version=1, observation_version=2, candidate_radius_px=3., patch_radius_px=2.,
    minimum_total_mass=.02, foreground_purity=.95, minimum_patch_mass=.005,
    depth_tolerance=.02, depth_spacing_multiplier=4., maximum_support_rows=8,
    bootstrap_max_error_px=1., ambiguity_gap_px=.25, forward_mass_relative_tolerance=5e-4,
    bootstrap_only=True, frozen_row_membership_and_weights=True)


@torch.no_grad()
def bootstrap_evidence(renderer, scene, cam, t, mask, queries, config):
    dynamic = scene.dynamic_at(t)
    full = compose(scene.static, [dynamic])
    xyz = dynamic.get_xyz
    h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], -1)@cam.full_proj_transform
    uv = ((h[:, :2]/h[:, 3:].clamp_min(1e-8)+1)*xyz.new_tensor([400, 225])-1)/2
    depth = (torch.cat([xyz, torch.ones_like(xyz[:, :1])], -1)@cam.world_view_transform)[:, 2]
    proposals = {}
    for track_id, target in queries:
        distance = (uv-uv.new_tensor(target)).norm(dim=-1)
        rows = torch.nonzero((h[:, 3] > 0) & torch.isfinite(h).all(-1)
                            & (distance <= config['candidate_radius_px'])).flatten()
        proposals[track_id] = (rows, distance)
    union = sorted({int(row) for rows, _ in proposals.values() for row in rows.tolist()})
    if not union:
        return {k:{} for k, _ in queries}, 0., 0
    offset = len(scene.static.get_xyz)
    colors = torch.zeros((len(full.get_xyz), 3), device=xyz.device)
    colors[torch.tensor(union, device=xyz.device)+offset] = 1.
    expected = renderer(cam, full, PIPE, xyz.new_zeros(3), 0., 0., 0., override_color=colors)['render'][0]
    summed = torch.zeros_like(expected)
    yy, xx = torch.meshgrid(torch.arange(225, device=xyz.device), torch.arange(400, device=xyz.device), indexing='ij')
    foreground = torch.as_tensor(mask, device=xyz.device)
    patches = {k: ((xx-target[0]).square()+(yy-target[1]).square() <= config['patch_radius_px']**2)
               for k, target in queries}
    records = {k:{} for k, _ in queries}
    requested = {row:[] for row in union}
    for track_id, (rows, _) in proposals.items():
        for row in rows.tolist():
            requested[row].append(track_id)
    for start in range(0, len(union), 3):
        rows = union[start:start+3]
        colors.zero_()
        colors[torch.tensor(rows, device=xyz.device)+offset, torch.arange(len(rows), device=xyz.device)] = 1.
        contribution = renderer(cam, full, PIPE, xyz.new_zeros(3), 0., 0., 0., override_color=colors)['render'][:len(rows)]
        summed += contribution.sum(0)
        for channel, row in enumerate(rows):
            total = float(contribution[channel].sum())
            fg = float(contribution[channel][foreground].sum())
            for track_id in requested[row]:
                records[track_id][row] = dict(valid=True,
                    distance=float(proposals[track_id][1][row]), depth=float(depth[row]),
                    total_mass=total, foreground_mass=fg,
                    patch_mass=float(contribution[channel][patches[track_id]].sum()),
                    homogeneous=h[row].cpu().double().tolist())
    error = float((summed-expected).abs().sum()/expected.sum().clamp_min(1))
    assert np.isfinite(error) and error < config['forward_mass_relative_tolerance'], error
    return records, error, len(union)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-surface-trajectory', action='store_true')
    parser.add_argument('--observation-version', type=int, choices=[2, 3], default=2)
    args = parser.parse_args()
    if not args.enable_surface_trajectory:
        raise SystemExit('Surface binding is opt-in')
    out = OUT if args.observation_version == 2 else OUT/'landmark_v3'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'bindings.json').exists(), 'Preserve completed binding result'
    torch.set_num_threads(4)
    verify_sources()
    assert sha(PARENT) == PARENT_SHA
    source_path = OUT/f'observation_v{args.observation_version}/image_tracks.json'
    source = read(source_path)
    assert source['parent_sha256'] == PARENT_SHA and source['training_only']
    renderer = setup()
    from ..union.model import UnionScene
    protocol = read(REPORT/'S3_union/protocol.json')
    scene = UnionScene(protocol)
    state = torch.load(PARENT, map_location='cpu', weights_only=False)
    scene.load_state_dict(state['model'])
    scene.requires_grad_(False)
    config = dict(CONFIG)
    config['observation_version'] = args.observation_version
    config['foreground_policy'] = 'original_yellow_union' if args.observation_version == 2 else 'yellow_orange_holefilled_envelope'
    config['depth_tolerance'] = max(config['depth_tolerance'], config['depth_spacing_multiplier']*float(scene.pools[0].spacing))
    config['parent_sha256'] = PARENT_SHA
    config['observation_sha256'] = sha(source_path)
    protocol_path = out/'binding_protocol.json'
    if protocol_path.exists():
        assert read(protocol_path) == config
    else:
        write(protocol_path, config)
    meta = read(DATA/'transforms_train.json')
    frames = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    frame_queries = {}
    evidence = {t['track_id']:{} for t in source['tracks']}
    for track in source['tracks']:
        for n, uv in zip(track['frames'][:3], track['uv_400'][:3]):
            frame_queries.setdefault(n, []).append((track['track_id'], uv))
    mass_checks = []
    for number, queries in sorted(frame_queries.items()):
        frame = frames[number]
        cam = camera(frame, meta, 400)
        rgba = image(frame, 400)
        mask = dynamic_mask(rgba) if args.observation_version == 2 else appearance_masks(rgba, 2, False)[0]
        records, error, count = bootstrap_evidence(renderer, scene, cam, frame['time'], mask, queries, config)
        for track_id, values in records.items():
            evidence[track_id][number] = values
        mass_checks.append(dict(frame=number, candidate_rows=count, forward_conservation_error=error))
        print('SURFACE_BOOTSTRAP', number, 'queries', len(queries), 'candidate_rows', count, 'mass_error', error, flush=True)
    bindings, rejected = [], []
    for track in source['tracks']:
        observations = [evidence[track['track_id']][n] for n in track['frames'][:3]]
        best, bad = choose_support(observations, track['uv_400'][:3], config)
        if best is None:
            rejected.append(dict(track_id=track['track_id'], time_band=track['time_band'], **bad))
        else:
            bindings.append(dict(track, **best))
    assert all(torch.equal(v.cpu(), state['model'][k]) for k, v in scene.state_dict().items())
    assert all(p.grad is None for p in scene.parameters()) and sha(PARENT) == PARENT_SHA
    verify_sources()
    result = dict(stage='R1', parent_sha256=PARENT_SHA, observation_sha256=sha(source_path),
        binding_protocol_sha256=sha(protocol_path), bindings=bindings, rejected=rejected,
        counts_by_time_band=dict(Counter(b['time_band'] for b in bindings)),
        rejection_counts=dict(Counter(r['reason'] for r in rejected)),
        complete_forward_mass_checks=mass_checks, all_model_tensors_unchanged=True,
        optimizer_updates=0, human_review_completed=False, split_pending=True,
        admission=False, bootstrap_only=True, heldout_used=False,
        physical_material_ID_GT=False, limitation='Model-conditional local support, not proved material identity')
    write(out/'bindings.json', result)
    write(out/'bootstrap_evidence.json', evidence)
    print('SURFACE_BINDING_COMPLETE', len(bindings), '/', len(source['tracks']), result['counts_by_time_band'],
          result['rejection_counts'], flush=True)


if __name__ == '__main__':
    main()
