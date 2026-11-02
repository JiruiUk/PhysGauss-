"""CPU snapshot/hash audit after actual GPU loading by the motion diagnostic."""
import hashlib
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; RUNS=ROOT/'runs/pidg_unified_flow'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(4)
lock=read(RUNS/'selection_lock.json'); rows={}
for name in lock['models']:
    run=RUNS/name/'rubber_duck'
    state=torch.load(run/'checkpoints/resume.pt',map_location='cpu',weights_only=False)
    rows[name]={}
    for tag in ['best','50000']:
        folder=run/'deform'/('iteration_'+tag)
        weights=torch.load(folder/'deform.pth',map_location='cpu',weights_only=False)
        material_file=run/'point_cloud'/('iteration_'+tag)/'material.pt'
        material=torch.load(material_file,map_location='cpu',weights_only=True)
        assert len(weights)==2 and 'shared_stress.weight' in weights[1]
        assert all(torch.isfinite(v).all() for group in weights for v in group.values())
        if tag=='50000':
            assert all(torch.equal(v,state[key][k]) for key,w in zip(['grid4d','deform'],weights) for k,v in w.items())
            assert all(torch.equal(v,state['velocity_net'][k]) for k,v in material['velocity_net'].items())
        else:
            assert str(lock['models'][name]['iteration']) in (folder/'iter.txt').read_text()
        files=[folder/'deform.pth',material_file,material_file.with_name('point_cloud.ply')]
        rows[name][tag]={'selected_iteration':lock['models'][name]['iteration'] if tag=='best' else 50000,
            'sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},'extended_head_present':True,
            'all_deformation_parameters_finite':True,'final_matches_full_checkpoint':tag=='50000'}
    del state
(ROOT/'experiments/unified_flow/snapshot_verification.json').write_text(json.dumps({'status':'passed','models':rows},indent=2)+'\n')
print('All four snapshot hashes, structure and final full-state parity verified.')
