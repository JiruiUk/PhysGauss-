"""One bounded native-visible-first search, preserving failed T3 history."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from ..i10_tracking.io import ROOT, DATA, read, write, sha
from ..i10_tracking.core import project_world
from ..train_static import PIPE, camera
from ..stage2.model import compose
from ..rubber_visibility.core import packing_colors
from ..joint_rgb.core import rng_state, restore_rng
from . import audit as a
from .core import patch_visibility, visibility_gate, neighborhood
from .visible_core import footprint_ids, choose, reciprocal_result

BASE = ROOT/'experiments/staged_reconstruction/RubberDuck_I10_track_neighborhood/visible'
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_track_neighborhood/visible/v1'
OUT = ROOT/'output/staged_reconstruction/RubberDuck_I10_track_neighborhood/visible/v1'


def install():
    for key in ['BASE', 'REPORT', 'RUNS', 'OUT']: setattr(a, key, globals()[key])
    a.r.RUNS = RUNS


def prepare():
    assert not a.stopped() and not (REPORT/'source_freeze.json').exists()
    assert read(REPORT/'unit_tests.json')['passed']
    files = {}
    for path in [BASE.parent/'v1/source_freeze.json']:
        for name, h in read(path)['files'].items():
            assert sha(ROOT/name) == h, name; files[name] = h
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [BASE/'plan.md', BASE/'protocol.json', BASE/'preregistration.json', ROOT/'tests/test_i10_track_visible.py']
    for path in paths: files[path.relative_to(ROOT).as_posix()] = sha(path)
    write(REPORT/'source_freeze.json', dict(files=files, optimizer_updates=0)); a.verify()
    print('T4_SOURCE_FROZEN', len(files), flush=True)


def collect_candidates(renderer, scene, tracks, xyz, projected, depth, meta, p):
    counts = {t['id']: [Counter(), Counter()] for t in tracks}
    fit = p['fit_observations']
    for index, frame in enumerate(meta['frames']):
        assert not a.stopped()
        assert np.array_equal(scene.dynamic_at(frame['time']).get_xyz.cpu().numpy(), xyz[index])
        active = [(t, index-t['start']) for t in tracks if t['start'] <= index < t['start']+12]
        if not active: continue
        package = scene.render(renderer, camera(frame, meta, 400), frame['time'])
        ids = package['gs_per_pixel'].cpu().numpy(); assert ids.shape[0] == p['native_ID_capacity']
        for track, local in active:
            rows = footprint_ids(ids, track['uv'][local], len(scene.static.get_xyz), len(scene.pools[0].get_xyz))
            counts[track['id']][0 if local in fit else 1].update(rows)
    candidates = {}
    for track in tracks:
        start = track['start']; sl = slice(start, start+12)
        error = np.linalg.norm(projected[sl]-np.array(track['uv'])[:, None], axis=-1)
        groups = []
        for counter, split in zip(counts[track['id']], [p['fit_observations'], p['held_observations']]):
            rms = np.sqrt(np.mean(error[split]**2, axis=0))
            groups.append([int(row) for row, count in counter.items() if count >= 4 and np.isfinite(rms[row]) and rms[row] <= 2.
                           and (depth[sl, row] > .01).all()])
        candidates[track['id']] = dict(fit=sorted(groups[0]), held=sorted(groups[1]), all=sorted(set(groups[0]+groups[1])))
    print('T4_VISIBLE_CANDIDATES', sum(len(c['all']) for c in candidates.values()), flush=True)
    return candidates


def contributions(renderer, scene, tracks, candidates, meta, p):
    obs = {t['id']: {r: [None]*12 for r in candidates[t['id']]['all']} for t in tracks}
    for index, frame in enumerate(meta['frames']):
        assert not a.stopped()
        active = [(t, index-t['start']) for t in tracks if t['start'] <= index < t['start']+12 and candidates[t['id']]['all']]
        if not active: continue
        ids = sorted({row for t, _ in active for row in candidates[t['id']]['all']})
        cam = camera(frame, meta, 400); bank = compose(scene.static, [scene.dynamic_at(frame['time'])])
        for begin in range(0, len(ids), 3):
            group = ids[begin:begin+3]
            colors = torch.from_numpy(packing_colors(len(scene.pools[0].get_xyz), len(scene.static.get_xyz), [[r] for r in group])).cuda()
            pkg = renderer(cam, bank, PIPE, scene.centers.new_zeros(3), 0., 0., 0., override_color=colors)
            alpha = pkg['alpha'].squeeze().cpu().numpy()
            for channel, row in enumerate(group):
                weight = pkg['render'][channel].cpu().numpy()
                for track, local in active:
                    if row in obs[track['id']]: obs[track['id']][row][local] = patch_visibility(weight, alpha, track['uv'][local], p)
        print('T4_VISIBLE_MEASURE', index+1, 120, len(ids), flush=True)
    return obs


@torch.no_grad()
def run(replay=False):
    a.verify(); assert not a.stopped(); torch.set_num_threads(4)
    assert read(REPORT/'native_check.json')['passed']
    p = read(BASE/'protocol.json'); source = read(ROOT/p['evidence'])
    tracks = [t for t in source['rows'] if t['stationarity']['label'] == 'motion_candidate']; assert len(tracks) == 222
    asset = np.load(ROOT/p['trajectories']); xyz, views, k = [asset[n] for n in ['xyz', 'views', 'k']]
    projected, depth = project_world(xyz, views, k)
    renderer, scene, opts, parent = a.load(); before = a.state(scene, opts)
    meta = read(DATA/'transforms_train.json')
    candidates = collect_candidates(renderer, scene, tracks, xyz, projected, depth, meta, p)
    obs = contributions(renderer, scene, tracks, candidates, meta, p); rows = []
    for track in tracks:
        sl = slice(track['start'], track['start']+12); values = obs[track['id']]
        assert all(all(v is not None for v in frames) for frames in values.values())
        forward = choose(track['uv'], xyz[sl], projected[sl], depth[sl], candidates[track['id']]['fit'], values,
                         p['fit_observations'], p['held_observations'], p)
        reverse = choose(track['uv'], xyz[sl], projected[sl], depth[sl], candidates[track['id']]['held'], values,
                         p['held_observations'], p['fit_observations'], p)
        reciprocal = reciprocal_result(forward, reverse, xyz[sl], projected[sl], p)
        visibility = visibility_gate(values[forward['row']], p) if 'row' in forward else dict(passed=False, reason='no_candidate')
        accepted = reciprocal['passed'] and visibility['passed']
        forward['accepted'] = bool(accepted)
        neighbors = neighborhood(xyz[sl], forward['row'], p) if 'row' in forward else dict(qualified=False, rows=[])
        rows.append(dict(id=track['id'], start=track['start'], band=track['band'], uv=track['uv'],
                         old_binding=track.get('binding'), binding=forward, swapped_binding=reverse,
                         reciprocal=reciprocal, visibility=visibility,
                         candidate_rows=candidates[track['id']], visibility_by_row={str(r):v for r,v in values.items()},
                         neighborhood=neighbors, accepted=bool(accepted)))
    accepted = [t for t in rows if t['accepted']]; graph = [t for t in accepted if t['neighborhood']['qualified']]
    def stats(values):
        bands = [sum(t['band'] == i for t in values) for i in range(3)]
        unique = len({t['binding']['row'] for t in values})
        return dict(tracks=len(values), distinct_Gaussian_rows=unique, by_band=bands,
                    passed=bool(len(values) >= p['minimum_bindings'] and min(bands) >= p['minimum_bindings_per_third'] and unique >= p['minimum_distinct_Gaussian_rows']))
    summary = dict(motion_candidates=len(tracks), candidate_row_pairs=sum(len(c['all']) for c in candidates.values()),
                   bindings=stats(accepted), graph=stats(graph), neighbor_target_rows=len({r for t in graph for r in t['neighborhood']['rows']}),
                   model_conditioned_not_3D_GT=True, optimizer_updates=0)
    a.r.assert_equivalent(before, a.state(scene, opts), 0, 0); a.verify()
    result = dict(rows=rows, summary=summary, full_state_no_mutation_exact=True, parent_camera_stack=parent['camera_stack'])
    if replay:
        assert result == read(REPORT/'audit.json')
        write(REPORT/'independent_replay.json', dict(passed=True, all222_native_source_and_binding_decisions_exact=True,
              all120_parent_3D_arrays_exact=True, full_state_no_mutation_exact=True, source_freeze_sha256=sha(REPORT/'source_freeze.json')))
    else:
        assert not (REPORT/'audit.json').exists(), 'Preserve completed result'
        write(REPORT/'audit.json', result); write(REPORT/'images.json', a.pictures(rows))
    print('T4_VISIBLE_BINDING_COMPLETE', summary, 'replay', replay, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-visible-binding', action='store_true')
    parser.add_argument('phase', choices=['prepare', 'native', 'audit', 'replay']); args = parser.parse_args()
    if not args.enable_visible_binding: raise SystemExit('T4 visible binding default off')
    install(); assert not a.stopped(); caller_rng = deepcopy(rng_state())
    try:
        with a.r.lock('audit.lock'):
            if args.phase == 'prepare': prepare()
            elif args.phase == 'native': a.qualification()
            else: run(args.phase == 'replay')
    finally: restore_rng(caller_rng)


if __name__ == '__main__': main()
