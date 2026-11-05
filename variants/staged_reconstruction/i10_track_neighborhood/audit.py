"""Opt-in native visibility and reciprocal-ID evidence; no updates."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from ..i10_tracking.io import ROOT, DATA, read, write, sha
from ..i10_tracking.core import project_world
from ..i10_track_train import runtime as parent_runtime
from ..rubber_visibility.worker import native, colors_for
from ..rubber_joint import runtime as r
from ..joint_rgb.core import rng_state, restore_rng
from ..state_recovery.runtime import graph_state
from ..train_static import camera
from .core import reciprocal, patch_visibility, visibility_gate, neighborhood

BASE = ROOT/'experiments/staged_reconstruction/RubberDuck_I10_track_neighborhood'
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_track_neighborhood/v1'
OUT = ROOT/'output/staged_reconstruction/RubberDuck_I10_track_neighborhood/v1'


def stopped():
    return any(p.exists() for p in [ROOT/'stop.request', RUNS/'stop.request', RUNS.parent/'stop.request', RUNS.parent.parent/'stop.request'])


def verify():
    for receipt in [BASE/'preregistration.json', REPORT/'source_freeze.json']:
        d = read(receipt)
        for name, h in d.get('files', d).items(): assert sha(ROOT/name) == h, name


def prepare():
    assert not stopped() and not (REPORT/'source_freeze.json').exists()
    assert read(REPORT/'unit_tests.json')['passed']
    p = read(BASE/'protocol.json'); files = {}
    inherited = read(ROOT/'experiments/staged_reconstruction/RubberDuck_I10_tracking/training/v1/source_freeze.json')
    for name, h in inherited.get('files', inherited).items():
        assert sha(ROOT/name) == h, name; files[name] = h
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [BASE/'protocol.json', BASE/'plan.md', BASE/'preregistration.json', ROOT/p['evidence'], ROOT/p['trajectories'],
              ROOT/p['old_protocol'], ROOT/'tests/test_i10_track_neighborhood.py',
              ROOT/'variants/staged_reconstruction/rubber_visibility/core.py',
              ROOT/'variants/staged_reconstruction/rubber_visibility/worker.py']
    for path in paths: files[path.relative_to(ROOT).as_posix()] = sha(path)
    write(REPORT/'source_freeze.json', dict(files=files, train_only=True, optimizer_updates=0)); verify()
    print('T3_SOURCE_AND_TRAIN_INPUTS_FROZEN', len(files), flush=True)


def state(scene, opts):
    return deepcopy(r.cpu_values(dict(model=scene.state_dict(), optimizers=[o.state_dict() for o in opts],
                                     bindings=r.bindings(scene, opts), physics_graphs=graph_state(scene), **rng_state())))


def load():
    p = read(BASE/'protocol.json'); assert sha(ROOT/p['parent']) == p['parent_sha256']
    renderer, scene, opts, _, parent = parent_runtime.load('rgb_only')
    r.assert_equivalent(scene.state_dict(), parent['model'], 0, 0)
    assert len(scene.pools[0].get_xyz) == 15507
    return renderer, scene, opts, parent


@torch.no_grad()
def qualification():
    verify(); assert not stopped(); torch.set_num_threads(4)
    renderer, scene, opts, parent = load(); before = state(scene, opts)
    meta = read(DATA/'transforms_train.json'); frame = meta['frames'][0]; cam = camera(frame, meta, 400)
    package = scene.render(renderer, cam, frame['time'])
    raw = native(renderer, scene, cam, frame['time'])
    assert all(torch.equal(package[k], raw[k]) for k in ['render', 'alpha', 'depth'])
    tracks = [t for t in read(ROOT/read(BASE/'protocol.json')['evidence'])['rows'] if t.get('binding', {}).get('accepted')]
    ids = [t['binding']['row'] for t in tracks[:3]]
    packed = native(renderer, scene, cam, frame['time'], colors_for(scene, [[i] for i in ids]))
    for channel, row in enumerate(ids):
        single = native(renderer, scene, cam, frame['time'], colors_for(scene, [[row]]))
        assert torch.allclose(packed['render'][channel], single['render'][0], atol=1e-7, rtol=1e-6)
        assert all(torch.equal(package[k], single[k]) for k in ['alpha', 'depth'])
    all_white = torch.ones((len(scene.static.get_xyz)+len(scene.pools[0].get_xyz), 3), device='cuda')
    white = native(renderer, scene, cam, frame['time'], all_white)
    assert torch.allclose(white['render'][0], package['alpha'].squeeze(), atol=1e-6, rtol=1e-6)
    r.assert_equivalent(before, state(scene, opts), 0, 0)
    assert all(p.grad is None for p in scene.parameters())
    write(REPORT/'native_check.json', dict(passed=True, RGB_alpha_D_exact=True, packed_unpacked_contribution=True,
          white_composite_equals_alpha=True, post_T_weights_not_used=True, full_state_no_mutation_exact=True, optimizer_updates=0))
    print('T3_NATIVE_QUALIFICATION_PASS', flush=True)


def pictures(results):
    p = read(BASE/'protocol.json'); meta = read(DATA/'transforms_train.json'); OUT.mkdir(parents=True, exist_ok=True)
    images = {}
    for start in p['fixed_review_starts']:
        rows = [r for r in results if r['start'] == start]
        sheet = Image.new('RGB', (1600, 750), '#151923'); draw = ImageDraw.Draw(sheet)
        for t in range(12):
            frame = meta['frames'][start+t]
            rgba = np.array(Image.open(DATA/(frame['file_path']+'.png')).convert('RGBA').resize((400, 225), Image.Resampling.BILINEAR))
            tile = Image.fromarray(np.uint8(rgba[..., :3].astype(float)*rgba[..., 3:4]/255)); d = ImageDraw.Draw(tile)
            for row in rows:
                x, y = row['uv'][t]; color = '#20ff70' if row['accepted'] else '#ff5050'
                d.ellipse((x-3, y-3, x+3, y+3), outline=color, width=1)
                d.text((x+4, y-6), str(row['id']), fill=color)
            px, py = 400*(t % 4), 250*(t//4)
            draw.text((px+3, py+3), f'train {start+t}: green=audited; red=unknown/rejected', fill='white')
            sheet.paste(tile, (px, py+23))
        path = OUT/f'bindings_{start:03d}.png'; sheet.save(path); images[path.relative_to(ROOT).as_posix()] = sha(path)
    return images


@torch.no_grad()
def audit(replay=False):
    assert not stopped(); verify(); torch.set_num_threads(4)
    assert read(REPORT/'native_check.json')['passed']
    p = read(BASE/'protocol.json'); original = read(ROOT/p['old_protocol'])
    evidence = read(ROOT/p['evidence']); tracks = [t for t in evidence['rows'] if t.get('binding', {}).get('accepted')]
    assert len(tracks) == 123
    asset = np.load(ROOT/p['trajectories']); xyz, views, k = [asset[n] for n in ['xyz', 'views', 'k']]
    predicted, depth = project_world(xyz, views, k)
    renderer, scene, opts, parent = load(); before = state(scene, opts)
    observations = {t['id']: [None]*12 for t in tracks}
    meta = read(DATA/'transforms_train.json')
    for index, frame in enumerate(meta['frames']):
        assert not stopped()
        assert np.array_equal(scene.dynamic_at(frame['time']).get_xyz.cpu().numpy(), xyz[index])
        active = [(t, index-t['start']) for t in tracks if t['start'] <= index < t['start']+12]
        if not active: continue
        cam = camera(frame, meta, 400)
        for begin in range(0, len(active), 3):
            group = active[begin:begin+3]
            packed = native(renderer, scene, cam, frame['time'], colors_for(scene, [[t['binding']['row']] for t, _ in group]))
            alpha = packed['alpha'].squeeze().cpu().numpy()
            for channel, (track, local) in enumerate(group):
                weight = packed['render'][channel].cpu().numpy()
                observations[track['id']][local] = patch_visibility(weight, alpha, track['uv'][local], p)
        print('T3_NATIVE_VISIBILITY', index+1, 120, flush=True)
    results = []
    for track in tracks:
        start = track['start']; sl = slice(start, start+12)
        assert all(o is not None for o in observations[track['id']])
        split = reciprocal(track, xyz[sl], predicted[sl], depth[sl], original, p)
        visible = visibility_gate(observations[track['id']], p)
        neighbors = neighborhood(xyz[sl], track['binding']['row'], p)
        results.append(dict(id=track['id'], start=start, band=track['band'], uv=track['uv'], original_binding=track['binding'],
                            reciprocal=split, visibility=visible, visibility_frames=observations[track['id']],
                            neighborhood=neighbors, accepted=bool(split['passed'] and visible['passed'])))
    accepted = [t for t in results if t['accepted']]
    counts = [sum(t['band'] == i for t in accepted) for i in range(3)]
    unique = len({t['original_binding']['row'] for t in accepted})
    gate = len(accepted) >= p['minimum_bindings'] and min(counts) >= p['minimum_bindings_per_third'] and unique >= p['minimum_distinct_Gaussian_rows']
    graph = [t for t in accepted if t['neighborhood']['qualified']]
    summary = dict(original_tracks=len(tracks), reciprocal_pass=sum(t['reciprocal']['passed'] for t in results),
                   native_visibility_pass=sum(t['visibility']['passed'] for t in results),
                   audited_tracks=len(accepted), distinct_Gaussian_rows=unique, by_band=counts,
                   reliable_binding_gate_passed=bool(gate), qualified_neighborhood_tracks=len(graph),
                   neighbor_target_rows=len({i for t in graph for i in t['neighborhood']['rows']}),
                   optimizer_updates=0, independent_3D_GT=False)
    r.assert_equivalent(before, state(scene, opts), 0, 0)
    assert all(p.grad is None for p in scene.parameters()); verify()
    result = dict(summary=summary, rows=results, full_state_no_mutation_exact=True,
                  parent_camera_stack=parent['camera_stack'], unknown_is_not_background=True)
    if replay:
        assert result == read(REPORT/'audit.json')
        write(REPORT/'independent_replay.json', dict(passed=True, all123_decisions_and_native_visibility_exact=True,
              all120_parent_3D_arrays_exact=True, full_state_no_mutation_exact=True, source_freeze_sha256=sha(REPORT/'source_freeze.json')))
    else:
        assert not (REPORT/'audit.json').exists(), 'Preserve completed audit'
        write(REPORT/'audit.json', result); write(REPORT/'images.json', pictures(results))
    print('T3_BINDING_AUDIT_COMPLETE', summary, 'replay', replay, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-binding-audit', action='store_true')
    parser.add_argument('phase', choices=['prepare', 'native', 'audit', 'replay']); args = parser.parse_args()
    if not args.enable_binding_audit: raise SystemExit('T3 audit default off')
    assert not stopped(); caller_rng = deepcopy(rng_state())
    r.RUNS = RUNS
    try:
        with r.lock('audit.lock'):
            if args.phase == 'prepare': prepare()
            elif args.phase == 'native': qualification()
            else: audit(args.phase == 'replay')
    finally: restore_rng(caller_rng)


if __name__ == '__main__': main()
