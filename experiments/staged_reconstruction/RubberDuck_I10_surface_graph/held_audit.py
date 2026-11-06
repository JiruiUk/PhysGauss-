"""Supplemental read-only held leakage stress and temporal coverage accounting."""
from pathlib import Path
import numpy as np
from variants.staged_reconstruction.i10_tracking.io import ROOT, read, write, sha
from variants.staged_reconstruction.i10_tracking.core import project_world
from variants.staged_reconstruction.i10_surface_graph.core import build_patches, build_edges


def main():
    b = Path(__file__).parent; p = read(b/'protocol.json'); h = read(b/'v1/hypotheses.json')
    health = read(b/'v1/initial_health.json')['controller']; evidence = read(ROOT/p['evidence'])
    with np.load(ROOT/p['trajectories']) as f:
        xyz = f['xyz']; uv, _ = project_world(xyz, f['views'], f['k'])
    tests = 0
    for t in evidence['rows']:
        if not t['accepted']:continue
        isolated = dict(rows=[t]); changed = xyz.copy(); projected = uv.copy()
        indices = np.array(p['held'])+t['start']
        # Different displacement for every row changes held relative lengths strongly.
        changed[indices, :, 0] += np.arange(xyz.shape[1], dtype=xyz.dtype)[None, :]+100
        projected[indices, :, 0] += np.arange(xyz.shape[1], dtype=uv.dtype)[None, :]+100
        a = build_patches(isolated, xyz, uv, p); c = build_patches(isolated, changed, projected, p)
        assert [(q['rows'],q['membership_weight']) for q in a] == [(q['rows'],q['membership_weight']) for q in c]
        ea, ec = build_edges(a, p), build_edges(c, p)
        assert [(q['i'],q['j'],q['owner']) for q in ea] == [(q['i'],q['j'],q['owner']) for q in ec]
        tests += 1
    rows = []
    for band in range(3):
        selected = [(q,s) for q,s in zip(h['patches'],health['patches']) if q['band']==band]
        edges = [(e,s) for e,s in zip(h['edges'],health['edges']) if h['patches'][e['owner']]['band']==band]
        rows.append(dict(temporal_third=band,anchors=sum(q['is_anchor'] for q,s in selected),
            proposed_neighbor_hypotheses=sum(not q['is_anchor'] for q,s in selected),
            active_neighbor_rows=len({q['rows'][0] for q,s in selected if not q['is_anchor'] and s['status']=='active'}),
            proposed_edges=len(edges),active_edges=sum(s['status']=='active' for e,s in edges)))
    write(b/'v1/held_audit.json',dict(passed=True,row_specific_held_corruption_windows=tests,
        nominations_and_fit_edges_unchanged=True,temporal_coverage=rows,
        supplemental_no_training_or_protocol_change=True,source_sha256=sha(Path(__file__))))
    print('L3_ROW_SPECIFIC_HELD_STRESS_PASS',tests,rows,flush=True)


if __name__=='__main__':main()
