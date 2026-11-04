"""Independently inspect saved graph predicates and frozen failure evidence."""
import numpy as np
from ..prepare_data import ROOT, read, write, sha
from .prepare import EXP, WORK, ANCHORS


def main():
    plan = read(EXP/'graph_protocol.json'); report = read(EXP/'graph_audit.json')
    for group in ['sources', 'inputs']:
        for name, expected in plan[group].items():
            assert sha(ROOT/name) == expected, name
    assert sha(EXP/'graph_protocol.json') == report['graph_protocol_sha256']
    assert sha(WORK/'graph.npz') == report['graph_sha256']
    a = dict(np.load(ANCHORS)); graph = dict(np.load(WORK/'graph.npz'))
    ids = graph['targets']; source = graph['sources']; weight = graph['spatial_weights']
    live = weight > 0; ref = graph['reference_indices']; n = len(a['opacity'])
    assert len(np.unique(ids)) == len(ids) == report['targets']
    assert (a['opacity'][ids] > .1).all() and live.sum(1).min() >= 3
    np.testing.assert_allclose(weight.sum(1), 1, rtol=1e-6)
    assert (weight >= 0).all() and np.isin(source[live], graph['keys']).all()
    assert (np.repeat(ids[:, None], source.shape[1], axis=1)[live] != source[live]).all()
    assert a['trusted'][np.broadcast_to(ref[:, None], source.shape)[live], source[live]].all()
    np.testing.assert_array_equal(graph['reference_target'], a['positions'][ref, ids])
    np.testing.assert_array_equal(graph['reference_sources'], a['positions'][ref[:, None], source])
    assert np.isfinite(graph['reference_covariance']).all()
    np.testing.assert_allclose(graph['reference_covariance'], graph['reference_covariance'].transpose(0,2,1), atol=1e-8)
    assert np.linalg.eigvalsh(graph['reference_covariance']).min() >= -1e-8
    position = a['positions']; quality = ((a['converged_mass_fraction'] >= .95)
                                               & (a['per_seed_support'].mean(0) > 1e-8))
    target = np.broadcast_to(ids[:, None], source.shape)[live]; neighbor = source[live]
    for start in range(0, len(target), 2048):
        i = target[start:start+2048]; j = neighbor[start:start+2048]
        common = quality[:, i] & a['trusted'][:, j]
        assert (common.sum(0) >= plan['construction']['common_min']).all()
        d = np.linalg.norm(position[:, i]-position[:, j], axis=-1)
        assert np.where(common, d, 0).max() <= plan['construction']['radius']
    reference = graph['reference_sources'].astype(np.float64)
    reference -= (reference*weight[..., None]).sum(1)[:, None]
    eig = np.linalg.eigvalsh((reference*weight[..., None]).transpose(0,2,1)@reference)
    assert (eig[:, -2]/np.maximum(eig[:, -1], 1e-12) >= plan['construction']['condition_min']).all()
    coverage = len(ids)/int((a['opacity'] > .1).sum())
    assert coverage == report['active_target_coverage'] and coverage < plan['active_target_coverage_min']
    assert report['passed'] is False and report['optimizer_updates'] == 0
    diagnosis = read(EXP/'coverage_diagnosis.json')
    assert sha(WORK/'coverage_diagnosis.npz') == diagnosis['arrays_sha256']
    assert sha(ROOT/'variants/staged_reconstruction/reliable_motion/diagnose.py') == diagnosis['script_sha256']
    masks = dict(np.load(WORK/'coverage_diagnosis.npz'))
    for name, mask in masks.items():
        row = diagnosis['definitions'][name]
        assert mask.sum(1).tolist() == [row[k] for k in ['targets_with_three_reference_neighbors',
            'targets_after_three_common_observations', 'targets_after_all_common_time_radius_before_rank_test']]
    assert '35 passed' in (EXP/'final_unit_tests.log').read_text(encoding='utf-8', errors='replace')
    write(EXP/'verification.json', {'passed': True, 'frozen_inputs_and_sources_verified': True,
        'all_saved_active_edges_common_observation_and_radius_predicates_verified': True,
        'reference_coordinates_covariance_and_noncollinearity_verified': True,
        'failure_gate_not_changed': True, 'coverage': coverage, 'unit_tests': 35,
        'graph_sha256': sha(WORK/'graph.npz'), 'script_sha256': sha(__file__),
        'training_started': False, 'test_RGB_read': False,
        'scope': 'Graph/failure evidence verification only; no new model or completeness claim'})
    print('RELIABLE_GRAPH_FAILURE_EVIDENCE_VERIFIED', len(ids), len(target), flush=True)


if __name__ == '__main__':
    main()
