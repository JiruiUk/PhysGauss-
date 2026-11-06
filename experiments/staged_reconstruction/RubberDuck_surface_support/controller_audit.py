"""Independent native controller replay for both arms, no optimizer updates."""
from copy import deepcopy
from pathlib import Path
import argparse
import torch
from variants.staged_reconstruction.i10_surface_support import runtime as a
from variants.staged_reconstruction.i10_surface_support.structure import refresh_surface
from variants.staged_reconstruction.i10_tracking.io import ROOT, read, write, sha
from variants.staged_reconstruction.i10_revocable_surface.core import transition


def run():
    r=a.install(True);r.verify();torch.set_num_threads(4);checks=[];usage={}
    for arm in a.CONFIG['arms'] if a.CONFIG else read(a.BASE/'protocol.json')['arms']:
        renderer,scene,opts,_,_=a.load(arm)
        for path in sorted((a.RUNS/'native_refresh'/arm).glob('model_*.json')):
            receipt=read(path);model=ROOT/receipt['model_path'];assert sha(model)==receipt['model_sha256']
            payload=torch.load(model,map_location='cpu',weights_only=False);scene.load_state_dict(payload['model'],strict=True)
            before=deepcopy(receipt['controller_before']);a.STATE=deepcopy(before)
            actual=a.measure(renderer,scene);after=receipt['controller_after']
            for i,(measurement,state) in enumerate(zip(actual,after['memories'])):
                assert measurement==state['measurement'],(arm,path.name,i)
                assert transition(before['memories'][i],measurement,a.CONFIG)==state
            surface=refresh_surface(renderer,scene,before['surface'],a.CONFIG) if arm=='surface_support' else before['surface']
            assert surface==after['surface'],(arm,path.name,'surface controller')
            assert payload['completed_updates']==before['calls']==receipt['summary']['completed_updates']
            checks.append(dict(arm=arm,completed_updates=before['calls'],memories_replayed=len(actual),
                               surface_nodes_replayed=len(surface['patches']) if arm=='surface_support' else 0,model_sha256=sha(model)))
        rows=read(a.RUNS/'formal'/arm/'usage.json');assert len(rows)==2000
        final=torch.load(a.RUNS/'formal'/arm/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert [q['index'] for q in rows]==final['sample_trace']
        assert all(all(not f%2 and f!=q['index'] for f in q['keyframes']) and len(q['keyframes'])<=4 for q in rows)
        if arm=='temporal_control':assert all(q['weighted_structure_loss']==0 for q in rows)
        else:assert all(q.get('coverage_groups',0)<=3 and (q['index']%2==0 or q.get('surface_patches',0)==0) for q in rows)
        usage[arm]=dict(steps_with_memory=sum(q['endpoint_pairs']>0 for q in rows),
            endpoint_pair_applications=sum(q['endpoint_pairs'] for q in rows),
            steps_with_structure=sum(q.get('surface_patches',0)>0 for q in rows),
            structural_patch_applications=sum(q.get('surface_patches',0) for q in rows),
            structural_edge_applications=sum(q.get('surface_edges',0) for q in rows),
            native_coverage_group_applications=sum(q.get('coverage_groups',0) for q in rows),
            mean_weighted_memory_loss=sum(q['weighted_memory_loss'] for q in rows)/len(rows),
            mean_weighted_structure_loss=sum(q['weighted_structure_loss'] for q in rows)/len(rows),
            final_history=final['controller']['history'])
    assert len(checks)==14;r.verify()
    write(a.REPORT/'controller_audit.json',dict(passed=True,all14_native_refreshes=checks,all4000_samples=True,
        fit_only_keyframes_and_structure_checked=True,arms=usage,source_sha256=sha(Path(__file__))))
    print('L5_CONTROLLER_REPLAY_PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--enable-controller-audit',action='store_true')
    if not p.parse_args().enable_controller_audit:raise SystemExit('Controller audit default off')
    run()
