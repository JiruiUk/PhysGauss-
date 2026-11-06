"""Independent-process native refresh replay and explicit decision accounting."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from variants.staged_reconstruction.i10_revocable_surface import runtime as a
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha


def independent_transition(before,after,p):
    for kind in ['patches','edges']:
        for old,new in zip(before[kind],after[kind]):
            m=new['measurement'];passed=m['kind']=='pass'
            good=old['pass_streak']+1 if passed else 0
            bad=old['fail_streak']+1 if m['kind']=='contradiction' else 0
            if passed:
                status='active' if old['status']=='active' or good>=p['passes_to_restore'] else old['status']
            else:status='revoked' if old['status']=='revoked' or bad>=p['failures_to_revoke'] else 'suspended'
            confidence=m['confidence'] if status=='active' else 0.
            assert (new['status'],new['pass_streak'],new['fail_streak'],new['confidence'])==(status,good,bad,confidence)
            if kind=='patches':
                frames=m['frames'];errors=np.array([q['error_px'] for q in frames])
                for q in frames:
                    assert q['passed']==bool(q['known'] and q['error_px']<=p['projection_max'] and q['ncc']>=p['ncc_min'])
                    if q['known']:assert q['share']>=p['visibility_min_share'] and min(q['observed_std'],q['rendered_std'])>=p['texture_std_255_min']
                if m['passed']:
                    fit=float(np.sqrt(np.mean(errors[p['fit']]**2)));held=float(np.sqrt(np.mean(errors[p['held']]**2)))
                    assert fit==m['fit_rms_px'] and held==m['held_rms_px']
                    bits=np.array([q['passed'] for q in frames]);assert bits.sum()>=p['min_valid_frames'] and bits[p['held']].sum()>=p['min_valid_held']
                    score=min(float(bits[p['fit']].mean()),float(bits[p['held']].mean()))*max(0.,1-max(fit,held)/p['projection_max'])
                    assert score==m['confidence']


def run():
    r=a.install(True);r.verify();torch.set_num_threads(4)
    for receipt in ['trace_freeze.json','evaluation_freeze.json']:
        for name,h in read(a.REPORT/receipt)['files'].items():assert sha(ROOT/name)==h,name
    renderer,scene,opts,_,parent=r.load('revocable_surface');records=[]
    directory=a.RUNS/'native_refresh/revocable_surface'
    paths=sorted(directory.glob('model_*.json'));assert len(paths)==7
    for expected,path in zip(range(250,2000,250),paths):
        record=read(path);model_path=ROOT/record['model_path'];assert sha(model_path)==record['model_sha256']
        payload=torch.load(model_path,map_location='cpu',weights_only=False)
        assert payload['model_only_not_resume'] and payload['completed_updates']==expected
        scene.load_state_dict(payload['model'],strict=True);a.STATE=deepcopy(record['controller_before'])
        assert a.STATE['calls']==expected
        summary=a.refresh(renderer,scene)
        assert summary==record['summary'];r.assert_equivalent(a.STATE,record['controller_after'],0,0)
        independent_transition(record['controller_before'],record['controller_after'],a.CONFIG)
        records.append(dict(completed_updates=expected,native_all122_replay_exact=True,
            model_sha256=record['model_sha256'],record_sha256=sha(path),summary=summary))
    final=a.RUNS/'formal/revocable_surface/checkpoint.pt';checkpoint_sha=sha(final)
    state=r.restore(final,scene,opts);saved=deepcopy(a.STATE)
    centers,measurements=a.measure(renderer,scene);assert a.STATE==saved and sha(final)==checkpoint_sha
    write(a.REPORT/'terminal_probe.json',dict(read_only=True,checkpoint_sha256=checkpoint_sha,
        measurements=measurements,passed=sum(m['passed'] for m in measurements),total=len(measurements),
        trained_controller_not_updated=True,independent_geometry_GT=False))
    arms={}
    for arm in a.CONFIG['arms']:
        path=a.RUNS/'formal'/arm/'checkpoint.pt';f=torch.load(path,map_location='cpu',weights_only=False)
        usage=read(path.parent/'usage.json');assert len(usage)==2000
        assert f['controller']['calls']==2000 and f['sample_trace']==[v['index'] for v in usage]
        history=f['controller']['history'];assert len(history)==(8 if arm=='revocable_surface' else 1)
        assert [v['completed_updates'] for v in history]==([0]+list(range(250,2000,250)) if arm=='revocable_surface' else [0])
        arms[arm]=dict(checkpoint_sha256=sha(path),updates=2000,history=history,
            samples_with_track=sum(v['tracks']>0 for v in usage),samples_with_edge=sum(v['edges']>0 for v in usage),
            total_patch_applications=sum(v['tracks'] for v in usage),total_edge_applications=sum(v['edges'] for v in usage),
            weighted_track_mean=float(np.mean([v['weighted_track_loss'] for v in usage])),
            weighted_local_mean=float(np.mean([v['weighted_local_loss'] for v in usage])))
    write(a.REPORT/'controller_audit.json',dict(passed=True,all7_native_refreshes_replayed=records,
        explicit_CPU_status_and_confidence_equations=True,all4000_usage_records_replayed=True,arms=arms,
        no_learned_confidence_or_deletion=True,terminal_probe_read_only=True,source_sha256=sha(Path(__file__))))
    print('L2_ALL_REFRESHES_NATIVE_AND_CONTROLLER_AUDIT_PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--enable-controller-audit',action='store_true')
    if not parser.parse_args().enable_controller_audit:raise SystemExit('Controller audit default off')
    run()
