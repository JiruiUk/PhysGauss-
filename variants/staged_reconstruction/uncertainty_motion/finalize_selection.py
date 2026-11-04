"""Audit recurring native-resolution train anchors before any graph training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
from ..prepare_data import ROOT, DATA, read, write, sha
from ..train_static import camera, PIPE
from ..stage2.model import compose
from ..stage2.train import image
from ..motion_integrity.prepare import PARENT, PARENT_SHA, load_parent
from .native_audit import EXP, OUT, detached_bank, snapshot_rng, assert_rng
from .observed_audit import model_digest, pixels
from .selection import select_seed, compare_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-uncertainty-diagnostic', action='store_true')
    args = parser.parse_args()
    if not args.enable_uncertainty_diagnostic:
        raise SystemExit('Anchor audit is opt-in')
    protocol_path = EXP/'observed_native120_protocol.json'; plan = read(protocol_path)
    for name, expected in plan['sources'].items():
        assert sha(ROOT/name) == expected, name
    result_path = EXP/plan['output_name']/'result.json'; raw = read(result_path)
    assert raw['protocol_sha256'] == sha(protocol_path)
    directory = EXP/'anchor_selection'; directory.mkdir(parents=True, exist_ok=True)
    assert not (directory/'assessment.json').exists(), 'Preserve selection audits'
    out = OUT/'anchor_selection'; out.mkdir(parents=True, exist_ok=True)
    records = []; arrays = []
    for record in raw['cases']:
        assert record['width'] == 1600
        path = OUT/plan['output_name']/f't{record["frame"]:04d}_w1600.npz'
        assert sha(path) == record['array_sha256']
        arrays.append(dict(np.load(path))); records.append(record)
    assert len(arrays) == 120
    source = torch.load(PARENT, map_location='cpu', weights_only=False)
    opacity = source['model']['pools.0._opacity'].sigmoid().numpy().reshape(-1)
    del source
    support = np.stack([a['support'] for a in arrays], axis=1)
    good = np.stack([a['converged_foreground_mass_fraction'] for a in arrays])
    positions = np.stack([a['positions'] for a in arrays])
    selection = plan['selection']
    seeds = [select_seed(support[i], good, opacity, positions, floor=plan['support_floor'],
        fraction=plan['top_fraction'], minimum_frames=selection['minimum_significant_frames'],
        voxel_size=selection['voxel_size']) for i in range(2)]
    jaccard = compare_seeds(seeds[0]['per_frame'], seeds[1]['per_frame'])
    stable = seeds[0]['anchors'] & seeds[1]['anchors']
    trusted = seeds[0]['per_frame'] & seeds[1]['per_frame']
    ids = np.flatnonzero(stable)
    current_counts = trusted[:, stable].sum(1)
    thirds = np.array([min((r['frame']-1)*3//200, 2) for r in records])
    recurring_per_third = [int((trusted[thirds == k][:, stable].sum(0) >= selection['minimum_significant_frames']).sum())
                          for k in range(3)]
    coverage = []
    for t, frame in enumerate(positions):
        visible = (good[t] >= .95) & (opacity > .1) & (support[:, t].mean(0) >= plan['support_floor'])
        distances = cKDTree(frame[stable]).query(frame[visible], k=1)[0] if len(ids) else np.full(int(visible.sum()), np.inf)
        coverage.append(float(np.mean(distances <= selection['coverage_radius'])) if len(distances) else 0.)
    gates = {'all_native_measurements_pass': raw['passed'],
        'independent_selection_jaccard': bool((jaccard >= selection['per_frame_seed_jaccard_min']).all()),
        'stable_anchor_count': len(ids) >= selection['stable_anchor_count_min'],
        'trusted_anchor_count_each_frame': bool((current_counts >= selection['trusted_anchors_each_frame_min']).all()),
        'recurring_in_each_temporal_third': min(recurring_per_third) >= selection['recurring_anchors_each_third_min'],
        'observed_model_support_neighbor_coverage': min(coverage) >= selection['qualified_support_neighbor_coverage_min']}
    np.savez_compressed(out/'anchors.npz', anchor_ids=ids, per_seed_support=support,
        converged_mass_fraction=good, positions=positions, frames=np.array([r['frame'] for r in records]),
        opacity=opacity, per_seed_selections=np.stack([s['per_frame'] for s in seeds]),
        per_seed_period=np.stack([s['period'] for s in seeds]), trusted=trusted, stable_anchors=stable)
    torch.set_num_threads(4)
    renderer, scene, optimizer, state, original = load_parent()
    del optimizer, state, original
    before_rng = snapshot_rng(); before_model = model_digest(scene)
    meta = read(DATA/'transforms_train.json')
    byid = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    rendered = []
    with torch.no_grad():
        for number in selection['visual_train_frames']:
            frame = byid[number]; cam = camera(frame, meta, 1600)
            bank = detached_bank(compose(scene.static, [scene.dynamic_at(frame['time'])]))
            rgba = image(frame, 1600)
            gt = torch.tensor(rgba[..., :3]*rgba[..., 3:4], device='cuda').permute(2, 0, 1)
            pred = scene.render(renderer, cam, frame['time'])['render']
            color = gt.new_zeros(len(bank.get_xyz), 3)
            offset = len(scene.static.get_xyz)
            color[offset:][torch.tensor(stable, device='cuda')] = 1
            anchor_mass = renderer(cam, bank, PIPE, gt.new_zeros(3), 0., 0., 0., override_color=color)['render']
            color[offset:] = 1
            full_mass = renderer(cam, bank, PIPE, gt.new_zeros(3), 0., 0., 0., override_color=color)['render']
            path = out/f't{number:04d}_recurring.png'
            sheet = Image.new('RGB', (1600, 330), '#151923'); draw = ImageDraw.Draw(sheet)
            for col, (label, value) in enumerate([('Actual train RGB', gt), ('S3-U unchanged RGB', pred),
                    ('Recurring anchors contribution', anchor_mass), ('All dynamic contribution', full_mass)]):
                sheet.paste(Image.fromarray(pixels(value)).resize((400, 225)), (400*col, 28))
                draw.text((400*col+6, 7), label, fill='white')
            draw.text((8, 275), f'frame {number}; observed input camera; anchor density is not surface completeness or identity truth', fill='white')
            sheet.save(path)
            rendered.append({'frame': number, 'sha256': sha(path),
                'anchor_compositing_mass_fraction': float(anchor_mass[0].sum()/full_mass[0].sum())})
    assert_rng(before_rng); assert model_digest(scene) == before_model
    assert sha(PARENT) == PARENT_SHA
    for name, expected in plan['sources'].items():
        assert sha(ROOT/name) == expected, name
    report = {'passed_for_graph_feasibility_only': all(gates.values()), 'gates': gates,
        'optimizer_updates': 0, 'test_RGB_read': False, 'native_train_frames': 120,
        'independent_seed_selection_jaccard_min': float(jaccard.min()),
        'independent_seed_selection_jaccard_median': float(np.median(jaccard)),
        'per_seed_anchor_counts': [int(s['anchors'].sum()) for s in seeds], 'stable_anchor_count': len(ids),
        'trusted_anchors_each_frame_min': int(current_counts.min()),
        'recurring_anchors_by_temporal_third': recurring_per_third,
        'model_support_neighbor_coverage_min': min(coverage),
        'model_support_neighbor_coverage_median': float(np.median(coverage)),
        'parent_sha256': PARENT_SHA, 'all_model_tensors_and_RNG_unchanged': True,
        'protocol_sha256': sha(protocol_path), 'raw_audit_sha256': sha(result_path),
        'anchor_arrays_sha256': sha(out/'anchors.npz'), 'native_rendered_train_figures': rendered,
        'scope': 'Recurring color support and graph feasibility, not depth, same-object identity or full surfaces; no model trained'}
    write(directory/'assessment.json', report)
    print('RECURRING_ANCHOR_AUDIT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
