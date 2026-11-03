"""Audit I4 at its rejected16k gate; never pretend it completed22k."""
import json
import random
import numpy as np
import torch
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha


def main():
    exp=ROOT/'experiments/staged_reconstruction/early_depth'
    folder=ROOT/'runs/pidg_staged_reconstruction/early_depth/formal/ordinal'
    source=ROOT/'runs/pidg_staged_reconstruction/S3_union/warmup/formal/initial.pt'
    original=torch.load(source,map_location='cpu',weights_only=False)
    initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    baseline=[json.loads(line)['frame_index'] for line in (source.parent/'sampling.jsonl').read_text().splitlines()]
    assert all(torch.equal(v,initial['model'][k]) for k,v in original['model'].items())
    assert initial['optimizer']==original['optimizer']
    assert len(rows)==final['iteration']==16000
    assert [row['global_step'] for row in rows]==list(range(1,16001))
    assert [row['frame_index'] for row in rows]==baseline[:16000]
    assert all(row['phase']=='warmup' and 'ordinal' in row and 'physics' not in row for row in rows)
    assert all(row['width']==(800 if row['global_step']<=12000 else 1600) for row in rows)
    rng=random.Random(); rng.setstate(original['rng_python']); stack=list(original['camera_stack'])
    for row in rows:
        if not stack: stack=rng.sample(range(120),120)
        assert row['frame_index']==stack.pop()
    assert final['rng_python']==rng.getstate() and final['camera_stack']==stack
    assert torch.equal(final['rng_torch'],original['rng_torch'])
    assert all(torch.equal(a,b) for a,b in zip(final['rng_cuda'],original['rng_cuda']))
    assert np.array_equal(final['rng_numpy'][1],original['rng_numpy'][1])
    assert all(v.isfinite().all() for v in final['model'].values())
    assert all(torch.equal(v,original['model'][k]) for k,v in final['model'].items() if k.startswith('static.'))
    for group in final['optimizer']['param_groups']:
        for index in group['params']:
            if group['name']=='PIDG_velocity_stress': assert index not in final['optimizer']['state']
            else:
                state=final['optimizer']['state'][index]; assert float(state['step'])==16000
                assert all(not torch.is_tensor(v) or v.isfinite().all() for v in state.values())
    assert sha(folder/'checkpoint.pt')==read(folder/'checkpoint.json')['sha256']
    assert final['protocol_sha256']==sha(exp/'protocol.json')
    admission=read(exp/'warmup_admission.json'); assert not admission['passed']
    assert not admission['numeric_gates']['static_preservation']
    for name,digest in read(exp/'source_freeze.json').items(): assert sha(ROOT/name)==digest,name
    write(exp/'training_verification_aborted.json',{
        'state_audit_passed':True,'actual_optimizer_updates':16000,'planned_updates':22000,
        'original_initial_state_Adam_and_sampler_exact':True,'static_tensors_frozen':True,
        'original_camera_and_width_schedule_exact':True,'physics_calls':0,
        'checkpoint_sha256':sha(folder/'checkpoint.pt'),'original_source_freeze_verified':True,
        'admitted_to_physics':False,'completeness_passed':False,'test_RGB_read':False,
        'stop_reason':'Original warmup static preservation gate failed;16k checkpoint retained'})
    images=ROOT/'output/staged_reconstruction/early_depth/warmup'
    names=['fixed86_t0086_native.png','fixed188_t0043_native.png','fixed188_t0146_native.png']
    write(exp/'visual_review.json',{
        'actual_visual_inspection_completed':True,'stage':'warmup16000_rejected',
        'images_reviewed':{name:sha(images/name) for name in names},
        'findings':'Fixed86/t86 retains yellow silhouettes, but fixed188/t43 andt146 show severe elongated ribbons, gaps and detached yellow fragments. Completeness clearly fails.',
        'completeness_passed':False,'full1200_scan_admitted':False,'test_RGB_read':False})
    print('I4_REJECTED_WARMUP_STATE_AUDIT_PASSED',flush=True)


if __name__=='__main__': main()
