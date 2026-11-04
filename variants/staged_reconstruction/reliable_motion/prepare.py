"""Build and qualify frozen reference patches from native train-only evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..motion_integrity.prepare import PARENT, PARENT_SHA, load_parent
from ..uncertainty_motion.native_audit import snapshot_rng, assert_rng
from ..uncertainty_motion.observed_audit import model_digest
from ..covariance_transport.core import covariance
from output.short_clip_diagnostic.camera_math import w2c
from .core import build_patches

EXP = ROOT/'experiments/staged_reconstruction/reliable_motion'
WORK = ROOT/'runs/pidg_staged_reconstruction/reliable_motion'
ANCHORS = ROOT/'output/staged_reconstruction/uncertainty_motion/anchor_selection/anchors.npz'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-reliable-motion', action='store_true')
    args = parser.parse_args()
    if not args.enable_reliable_motion:
        raise SystemExit('Reliable motion preparation is opt-in')
    plan = read(EXP/'graph_protocol.json')
    for name, expected in plan['inputs'].items():
        assert sha(ROOT/name) == expected, name
    for name, expected in plan['sources'].items():
        assert sha(ROOT/name) == expected, name
    assert not (EXP/'graph_audit.json').exists(), 'Preserve graph qualifications'
    a = np.load(ANCHORS)
    assert sha(ANCHORS) == read(ROOT/'experiments/staged_reconstruction/uncertainty_motion/anchor_selection/assessment.json')['anchor_arrays_sha256']
    graph = build_patches(a['positions'], a['per_seed_support'].mean(0), a['converged_mass_fraction'],
        a['opacity'], a['anchor_ids'], a['trusted'], **plan['construction'])
    coverage = len(graph['targets'])/int((a['opacity'] > .1).sum())
    print('RELIABLE_PATCH_GEOMETRY', len(graph['targets']), coverage, flush=True)
    torch.set_num_threads(4)
    renderer, scene, optimizer, state, original = load_parent()
    del renderer, optimizer, state, original
    before_rng = snapshot_rng(); before_model = model_digest(scene)
    frames = read(DATA/'transforms_train.json')['frames']
    assert [int(Path(f['file_path']).stem) for f in frames] == a['frames'].tolist()
    cov = np.zeros((len(graph['targets']), 3, 3), np.float32)
    with torch.no_grad():
        for t, frame in enumerate(frames):
            rows = np.flatnonzero(graph['reference_indices'] == t)
            if not len(rows):
                continue
            dynamic = scene.dynamic_at(frame['time'])
            xyz = dynamic.get_xyz.cpu().numpy()
            assert np.array_equal(xyz, a['positions'][t]), 'Bind covariance to the exact audited parent position'
            current = covariance(dynamic.get_scaling, dynamic.get_rotation)
            cov[rows] = current[graph['targets'][rows]].cpu().numpy()
    graph['reference_covariance'] = cov
    graph['camera_rotation'] = np.stack([w2c(f['transform_matrix'], 'metadata_cv')[:3,:3] for f in frames]).astype(np.float32)
    graph['frame_ids'] = a['frames']; graph['times'] = np.array([f['time'] for f in frames])
    lookup = np.zeros(a['positions'].shape[1], np.int64)
    lookup[graph['keys']] = np.arange(len(graph['keys'])); graph['key_lookup'] = lookup
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK/'graph.npz', **graph)
    assert_rng(before_rng); assert model_digest(scene) == before_model and sha(PARENT) == PARENT_SHA
    for name, expected in plan['sources'].items():
        assert sha(ROOT/name) == expected, name
    report = {'passed': coverage >= plan['active_target_coverage_min'],
        'targets': len(graph['targets']), 'active_parent_points': int((a['opacity'] > .1).sum()),
        'active_target_coverage': coverage, 'candidate_directed_pairs': int(graph['pair_candidate_count']),
        'qualified_directed_pairs': int(graph['qualified_pair_count']),
        'common_qualified_frame_quantiles': graph['pair_common_count_quantiles'].tolist(),
        'reference_condition_quantiles': np.quantile(graph['condition'], [0,.1,.5,1]).tolist(),
        'actual_neighbors_quantiles': np.quantile((graph['spatial_weights']>0).sum(1), [0,.1,.5,1]).tolist(),
        'reference_frame_coverage': len(np.unique(graph['reference_indices'])),
        'graph_sha256': sha(WORK/'graph.npz'), 'graph_protocol_sha256': sha(EXP/'graph_protocol.json'),
        'original_model_and_RNG_unchanged': True, 'optimizer_updates': 0, 'test_RGB_read': False,
        'scope': 'Conditional graph support, not same-object/material correspondence or geometry GT'}
    write(EXP/'graph_audit.json', report)
    print('RELIABLE_PATCH_GRAPH_AUDIT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
