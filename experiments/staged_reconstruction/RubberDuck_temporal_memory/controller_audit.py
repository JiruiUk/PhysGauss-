"""Independent native memory/transition replay; no optimizer updates."""
from copy import deepcopy
from pathlib import Path
import argparse
import torch
from variants.staged_reconstruction.i10_temporal_memory import runtime as a
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha
from variants.staged_reconstruction.i10_revocable_surface.core import transition

def run():
    r=a.install(True);r.verify();torch.set_num_threads(4)
    renderer,scene,opts,_,_=a.load('temporal_memory')
    checks=[]
    for path in sorted((a.RUNS/'native_refresh/temporal_memory').glob('model_*.json')):
        receipt=read(path);model=ROOT/receipt['model_path'];assert sha(model)==receipt['model_sha256']
        payload=torch.load(model,map_location='cpu',weights_only=False)
        scene.load_state_dict(payload['model'],strict=True)
        before=deepcopy(receipt['controller_before']);a.STATE=deepcopy(before)
        actual=a.measure(renderer,scene);expected=receipt['controller_after']['memories']
        for i,(measurement,state) in enumerate(zip(actual,expected)):
            assert measurement==state['measurement'],(path.name,i)
            assert transition(before['memories'][i],measurement,a.CONFIG)==state,(path.name,i)
        assert payload['completed_updates']==before['calls']==receipt['summary']['completed_updates']
        checks.append(dict(completed_updates=before['calls'],memories_replayed=len(actual),model_sha256=sha(model)))
    assert len(checks)==7
    usage={}
    for arm in a.CONFIG['arms']:
        rows=read(a.RUNS/'formal'/arm/'usage.json');assert len(rows)==2000
        final=torch.load(a.RUNS/'formal'/arm/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert [q['index'] for q in rows]==final['sample_trace']
        assert all(all(not f%2 and f!=q['index'] for f in q['keyframes']) and len(q['keyframes'])<=4 for q in rows)
        if arm=='rgb_control':assert all(q['weighted_memory_loss']==0 and q['endpoint_pairs']==0 and q['memories']==0 for q in rows)
        usage[arm]=dict(steps_with_memory=sum(q['endpoint_pairs']>0 for q in rows),
            total_endpoint_pairs=sum(q['endpoint_pairs'] for q in rows),total_memory_applications=sum(q['memories'] for q in rows),
            mean_weighted_memory_loss=sum(q['weighted_memory_loss'] for q in rows)/len(rows),
            final_history=final['controller']['history'])
    r.verify();write(a.REPORT/'controller_audit.json',dict(passed=True,all7_native_refreshes=checks,all4000_samples=True,
        fit_only_keyframes_checked=True,arms=usage,source_sha256=sha(Path(__file__))))
    print('L4_CONTROLLER_REPLAY_PASS',usage,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--enable-memory-audit',action='store_true')
    if not p.parse_args().enable_memory_audit:raise SystemExit('Memory replay default off')
    run()
