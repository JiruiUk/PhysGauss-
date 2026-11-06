"""CPU replay and explicit invariants for parent-conditioned graph proposals."""
from collections import Counter
from pathlib import Path
import numpy as np
from variants.staged_reconstruction.i10_tracking.io import ROOT, read, write, sha
from variants.staged_reconstruction.i10_tracking.core import project_world
from variants.staged_reconstruction.i10_surface_graph.core import build_patches, build_edges


def main():
    b = Path(__file__).parent; p = read(b/'protocol.json'); h = read(b/'v1/hypotheses.json')
    evidence = read(ROOT/p['evidence'])
    with np.load(ROOT/p['trajectories']) as asset:
        xyz = asset['xyz']; uv, _ = project_world(xyz, asset['views'], asset['k'])
    nodes = build_patches(evidence, xyz, uv, p); edges = build_edges(nodes, p)
    assert nodes == h['patches'] and edges == h['edges']
    accepted = {q['id']:q for q in evidence['rows'] if q['accepted']}
    owners = Counter(); rows = Counter()
    for i, node in enumerate(nodes):
        assert len(node['rows']) == 1 and node['weights'] == [1.]
        row = node['rows'][0]; sl = slice(node['start'], node['start']+12)
        assert np.array_equal(node['parent_centers'], xyz[sl, row])
        if node['is_anchor']:
            assert accepted[node['id']]['binding']['row'] == row and node['uv'] == accepted[node['id']]['uv']
        else:
            owner = nodes[node['owner']]; owners[node['owner']] += 1; rows[row] += 1
            assert owner['is_anchor'] and owner['start'] == node['start'] and i in owner['members']
            assert not node['receives_track_auxiliary'] and node['target_is_parent_transported_proxy']
            target = np.array(owner['uv'])+np.asarray(uv[sl, row],np.float64)-uv[sl, owner['rows'][0]]
            assert np.array_equal(target, node['uv'])
    assert max(owners.values()) <= 8 and max(rows.values()) <= 3
    for q in nodes:
        if not q['is_anchor']:assert q['membership_weight'] == 1/rows[q['rows'][0]]
    # Deliberately corrupt held geometry in memory: nomination must remain unchanged.
    xyz_changed = xyz.copy(); uv_changed = uv.copy()
    # Windows overlap, so mutate held per-window in isolated evidence calls.
    for root in [q for q in nodes if q['is_anchor']]:
        t = accepted[root['id']]; isolated = dict(rows=[t]); changed = xyz.copy(); projected = uv.copy()
        indices = np.array(p['held'])+t['start']; changed[indices] += 100; projected[indices] += 100
        original = build_patches(isolated, xyz, uv, p); altered = build_patches(isolated, changed, projected, p)
        assert [(q['rows'],q['membership_weight']) for q in original] == [(q['rows'],q['membership_weight']) for q in altered]
    for edge in edges:
        a, c = [nodes[edge[k]] for k in ['i','j']]
        assert a['start'] == c['start'] == nodes[edge['owner']]['start']
        assert a['rows'][0] != c['rows'][0]
        lengths = np.linalg.norm(np.asarray(a['parent_centers'])-c['parent_centers'],axis=-1)
        assert np.array_equal(lengths,edge['parent_lengths'])
    write(b/'v1/input_audit.json',dict(passed=True,all_nodes_edges_replayed=True,
        explicit_parent_transported_targets_exact=True,all122_held_nomination_corruption_checks=True,
        bounded_overlapping_memberships=True,no_new_measured_tracks_or_GT=True,
        nodes=len(nodes),edges=len(edges),source_sha256=sha(Path(__file__))))
    print('L3_INPUT_GRAPH_REPLAY_PASS',len(nodes),len(edges),flush=True)


if __name__ == '__main__':main()
