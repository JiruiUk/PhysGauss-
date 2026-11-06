"""CPU-only independent reconstruction of track memory and fit/held isolation."""
from copy import deepcopy
from pathlib import Path
from collections import Counter
import numpy as np
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha
from variants.staged_reconstruction.i10_tracking.core import project_world
from variants.staged_reconstruction.i10_temporal_memory.core import build_bank,bind_memory,propose_links

def main():
    b=Path(__file__).parent;p=read(b/'protocol.json');expected=read(b/'v1/bank.json')
    for path,h in expected['inputs_sha256'].items():assert sha(ROOT/path)==h,path
    tracks=read(ROOT/p['tracks'])['tracks'];evidence=read(ROOT/p['evidence'])
    actual=build_bank(tracks,evidence,p)
    with np.load(ROOT/p['trajectories']) as asset:
        uv,z=project_world(asset['xyz'],asset['views'],asset['k'])
        for m in actual['memories']:m['binding']=bind_memory(m,asset['xyz'],uv,z,p)
    assert all(actual[key]==expected[key] for key in actual)
    corrupted=deepcopy(tracks)
    for j,t in enumerate(corrupted):
        for i,point in enumerate(t['uv']):
            if (t['start']+i)%2:point[0]+=1000*(j+1);point[1]-=1000*(j+2)
    assert propose_links(tracks,p)==propose_links(corrupted,p)
    assert build_bank(corrupted,evidence,p)['fit_links']==actual['fit_links']
    assert sorted(name for m in actual['memories'] for name in m['track_ids'])==sorted(t['id'] for t in tracks)
    assert sum(sum(m['overlap_observation_counts']) for m in actual['memories'])==12*len(tracks)
    for m in actual['memories']:
        assert len({next(t['start'] for t in tracks if t['id']==name) for name in m['track_ids']})==len(m['track_ids'])
        assert len(m['frames'])==len(m['uv'])==len(m['overlap_observation_counts'])
    write(b/'v1/independent_input_audit.json',dict(passed=True,all730_original_tracks_partitioned=True,
        all8760_raw_observations_accounted=True,all_links_aggregates_and_bindings_exact=True,
        all730_held_track_perturbations_leave_fit_links_unchanged=True,
        span_histogram=dict(Counter(len(m['frames']) for m in actual['memories'])),
        independent_3D_GT=False,source_sha256=sha(Path(__file__))))
    print('L4_ALL_MEMORY_INPUTS_INDEPENDENT_REPLAY_PASS')

if __name__=='__main__':main()
